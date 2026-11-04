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

//! Hardware (HOT) GPU rasterization path.
//!
//! When [`codevar_truetype_core::RasterFlags::HOT`] is requested, the
//! gray rasterizer first tries this module: the glyph outline is
//! flattened on the CPU into a list of straight line segments (four
//! `f32` pixel coordinates per segment), uploaded to an OpenCL device,
//! and rasterized by the `font_raster` kernel — a 2x2 supersampled
//! PNPOLY winding scan compiled at build time by `codevar-oclc` and
//! embedded as SPIR-V. The kernel writes one coverage byte per pixel in
//! the same 0..=255 `PixelMode::Gray` convention the CPU scan converter
//! produces; only non-zero pixels are copied into the target bitmap so
//! the "leave untouched what the outline does not cover" contract of
//! FreeType's rasterizer holds on both paths.
//!
//! Everything here is best-effort: any failure — no OpenCL loader, no
//! GPU device, a driver that rejects the SPIR-V image, an allocation
//! error — is logged and reported as "not handled" so the caller falls
//! back to the CPU scan converter, which always produces a result. The
//! OpenCL pipeline (device, context, queue, program, kernel) is built
//! once on first use and cached for the lifetime of the process.
//!
//! The default path ends in host memory: the coverage is read back out of
//! the device and blitted into a [`Bitmap`], which is what a caller that
//! keeps its glyph cache in a `Vec` needs. A renderer whose cache lives on
//! the GPU does not, and [`attach_shared_coverage`] plus
//! [`try_render_shared`] give it the zero-copy shape of the same render:
//! the caller allocates the coverage target through its own Vulkan
//! device, imports it here as memory OpenCL may write, and the kernel
//! lands the glyph in it directly. See [`codevar_ocl_interop`] for what
//! makes that exchange sound — the short version is that the allocation
//! stays owned by Vulkan and OpenCL only touches it inside an acquire
//! window, which [`enqueue_shared`] opens and closes around one render.

use alloc::vec;
use alloc::vec::Vec;
use codevar_logger::{log_debug, log_warn};
use codevar_ocl::{Arg, Buffer, CommandQueue, Context, DeviceKind, Kernel, Program, Runtime};
use codevar_ocl_interop::{Error as InteropError, MemoryHandle, SharedBuffer};
use codevar_oclc::kernels;
use codevar_truetype_core::{Bitmap, OUTLINE_EVEN_ODD_FILL, Outline, TtResult, Vector};
use spin::Mutex;

use crate::decompose::{Decomposer, decompose};

/// Outline coordinates are 26.6 fixed point (1/64 pixel).
const COORD_SCALE: f32 = 1.0 / 64.0;
/// Maximum distance between a curve and its chord, in pixels, at which
/// flattening stops subdividing.
const FLATNESS: f32 = 0.1;
/// Hard cap on the subdivision depth of a single curve (guards against
/// pathological control points; at most `2^MAX_DEPTH` segments).
const MAX_DEPTH: u32 = 8;

/// Why the HOT pipeline could not be built.
enum PipelineError {
    /// No platform exposed an available GPU device.
    NoGpuDevice,
    /// An OpenCL call failed.
    OpenCl(codevar_ocl::Error),
}

impl From<codevar_ocl::Error> for PipelineError {
    #[inline]
    fn from(error: codevar_ocl::Error) -> Self {
        Self::OpenCl(error)
    }
}

impl core::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoGpuDevice => f.write_str("no available GPU device was exposed by any platform"),
            Self::OpenCl(error) => write!(f, "OpenCL call failed: {error}"),
        }
    }
}

/// Glyph coverage left in an allocation shared with another API.
///
/// A region always starts at byte zero of the attached allocation:
/// `font_raster` indexes `coverage[y * width + x]`, so one render occupies
/// [`CoverageRegion::size`] bytes from the start of the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoverageRegion {
    /// Covered width in pixels.
    pub width: usize,
    /// Covered height in pixels.
    pub height: usize,
}

impl CoverageRegion {
    /// Bytes of the shared allocation this region occupies, or `None` when
    /// the dimensions overflow — which a caller reads as "this glyph cannot
    /// be rasterized device-side".
    #[must_use]
    pub const fn size(self) -> Option<usize> {
        self.width.checked_mul(self.height)
    }
}

/// Why a shared coverage allocation could not be attached.
#[derive(Debug)]
pub enum AttachError {
    /// No HOT pipeline can be built on this machine, so there is no OpenCL
    /// context to import the allocation into.
    NoPipeline,
    /// The driver refused the import.
    Interop(InteropError),
}

impl From<InteropError> for AttachError {
    #[inline]
    fn from(error: InteropError) -> Self {
        Self::Interop(error)
    }
}

impl core::fmt::Display for AttachError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoPipeline => f.write_str("no HOT pipeline exists to import an allocation into"),
            Self::Interop(error) => write!(f, "the shared allocation was refused: {error}"),
        }
    }
}

/// A ready-to-enqueue HOT pipeline: the queue owns the context and
/// device, the kernel owns the built program, so this pair pins
/// everything the render path needs.
struct GpuPipeline {
    /// In-order command queue for the cached device.
    queue: CommandQueue,
    /// The `font_raster` kernel bound to the embedded SPIR-V program.
    kernel: Kernel,
    /// Coverage target imported from another API's allocation, once a
    /// caller attaches one; nothing is rasterized device-side without it.
    coverage: Option<SharedBuffer>,
}

/// Process-wide pipeline state, built lazily on the first HOT render.
enum PipelineState {
    /// Nothing attempted yet.
    Uninit,
    /// A build attempt failed; never touch the driver again.
    Unavailable,
    /// The pipeline is ready and reused for every subsequent render.
    Ready(GpuPipeline),
}

/// The cached pipeline; the mutex also serializes concurrent HOT
/// renders, which matches the serialized nature of a single queue.
static PIPELINE: Mutex<PipelineState> = Mutex::new(PipelineState::Uninit);

/// Builds the OpenCL pipeline
///
/// # Errors
///
/// Returns [`PipelineError::NoGpuDevice`] when no platform exposes an
/// available GPU, and the underlying [`codevar_ocl::Error`] for any
/// driver failure (including a rejected SPIR-V image).
fn build_pipeline() -> Result<GpuPipeline, PipelineError> {
    let runtime = Runtime::load()?;
    let mut selected = None;
    for platform in runtime.platforms() {
        let Ok(devices) = platform.devices(DeviceKind::Gpu) else {
            continue;
        };
        for device in devices {
            if device.available().unwrap_or(false) {
                selected = Some(device);
                break;
            }
        }
        if selected.is_some() {
            break;
        }
    }
    let device = selected.ok_or(PipelineError::NoGpuDevice)?;
    let context = Context::new(&device)?;
    let queue = CommandQueue::new(&context, &device)?;
    let program = Program::from_il(&context, kernels::truetype::RASTER_SPIRV)?;
    program.build("")?;
    let kernel = program.kernel("font_raster")?;
    Ok(GpuPipeline {
        queue,
        kernel,
        coverage: None,
    })
}

/// Runs `f` with exclusive access to the cached pipeline, initializing
/// it on first use. Returns `None` when the pipeline is unavailable;
/// the initialization failure is logged exactly once.
fn with_pipeline<R>(f: impl FnOnce(&mut GpuPipeline) -> R) -> Option<R> {
    let mut state = PIPELINE.lock();
    if matches!(*state, PipelineState::Uninit) {
        match build_pipeline() {
            Ok(pipeline) => *state = PipelineState::Ready(pipeline),
            Err(error) => {
                log_warn!("the hot rasterizer is unavailable, using the CPU path: {error}");
                *state = PipelineState::Unavailable;
            }
        }
    }
    match &mut *state {
        PipelineState::Ready(pipeline) => Some(f(pipeline)),
        PipelineState::Uninit | PipelineState::Unavailable => None,
    }
}

/// `true` when a HOT GPU pipeline can be built on this machine.
///
/// Attempts the one-time initialization if it has not run yet; a
/// machine without a usable OpenCL GPU simply returns `false` (and
/// caches that answer). Useful for tests and for callers that want to
/// know whether [`codevar_truetype_core::RasterFlags::HOT`] will take
/// the GPU path or fall back to the CPU scan converter.
#[must_use]
pub fn gpu_available() -> bool {
    with_pipeline(|_| ()).is_some()
}

/// Attaches an allocation exported by another API as the coverage target of
/// the HOT pipeline.
///
/// `handle` is the descriptor a Vulkan device produced for its own
/// `VkDeviceMemory` (see [`codevar_ocl_interop::Vulkan::export_memory`]) and
/// `size` is how many bytes of it this module may write; the allocation is
/// imported as a buffer of the pipeline's context, so after this call
/// [`try_render_shared`] rasterizes glyphs into memory that the exporting
/// device still owns. Attaching is a one-time setup cost paid when the
/// renderer already holds both drivers and their devices have been paired —
/// [`codevar_ocl_interop::Identity::pairs_with`] answers that question.
///
/// The descriptor is only read during the import and this crate never closes
/// it, so the caller closes it after this returns, whichever way it returned.
/// Attaching again replaces the previous target and releases that buffer;
/// [`try_render`] and the CPU path keep working either way.
///
/// # Errors
///
/// [`AttachError::NoPipeline`] when no GPU pipeline can be built here, and
/// [`AttachError::Interop`] when the driver refuses the handle — which is
/// also the answer on a context without `cl_khr_external_memory`, for a
/// handle another device exported, or for a `size` beyond the allocation.
pub fn attach_shared_coverage(handle: MemoryHandle, size: usize) -> Result<(), AttachError> {
    with_pipeline(|pipeline| {
        SharedBuffer::import(pipeline.queue.context(), size, handle).map(|shared| {
            pipeline.coverage = Some(shared);
        })
    })
    .ok_or(AttachError::NoPipeline)?
    .map_err(AttachError::Interop)
}

/// Detaches the shared coverage target, handing the allocation back to the
/// API that exported it.
///
/// A no-op when nothing is attached. Call it with no render in flight: the
/// imported buffer is what the kernel writes into, so dropping it mid-render
/// removes the target from under a command the queue may not have run yet.
pub fn detach_shared_coverage() {
    let _ = with_pipeline(|pipeline| pipeline.coverage = None);
}

/// `true` while a shared coverage target is attached, which is exactly when
/// [`try_render_shared`] can succeed.
#[must_use]
pub fn shared_coverage_attached() -> bool {
    with_pipeline(|pipeline| pipeline.coverage.is_some()).unwrap_or(false)
}

/// Collects outline segments as `f32` pixel-coordinate edges.
///
/// Each emitted edge appends four floats — x1, y1, x2, y2 — to
/// [`EdgeBuilder::edges`], the layout the `font_raster` kernel reads.
struct EdgeBuilder {
    /// Flat edge list, four floats per segment, contour order.
    edges: Vec<f32>,
    /// The end point of the most recently emitted segment.
    current: [f32; 2],
}

impl EdgeBuilder {
    /// Emits a straight segment from `current` to `to`.
    #[inline]
    fn push(&mut self, to: [f32; 2]) {
        self.edges
            .extend_from_slice(&[self.current[0], self.current[1], to[0], to[1]]);
        self.current = to;
    }

    /// Flattens a quadratic Bézier into line segments.
    fn flatten_quad(&mut self, control: [f32; 2], to: [f32; 2]) {
        let start = self.current;
        self.quad(start, control, to, 0);
        self.current = to;
    }

    /// Recursive de Casteljau subdivision of a quadratic segment.
    fn quad(&mut self, p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], depth: u32) {
        if depth >= MAX_DEPTH || point_segment_distance(p1, p0, p2) <= FLATNESS {
            self.edges
                .extend_from_slice(&[p0[0], p0[1], p2[0], p2[1]]);
            return;
        }
        let p01 = midpoint(p0, p1);
        let p12 = midpoint(p1, p2);
        let mid = midpoint(p01, p12);
        self.quad(p0, p01, mid, depth + 1);
        self.quad(mid, p12, p2, depth + 1);
    }

    /// Flattens a cubic Bézier into line segments.
    fn flatten_cubic(&mut self, control1: [f32; 2], control2: [f32; 2], to: [f32; 2]) {
        let start = self.current;
        self.cubic(start, control1, control2, to, 0);
        self.current = to;
    }

    /// Recursive de Casteljau subdivision of a cubic segment.
    fn cubic(&mut self, p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], depth: u32) {
        if depth >= MAX_DEPTH
            || (point_segment_distance(p1, p0, p3) <= FLATNESS
                && point_segment_distance(p2, p0, p3) <= FLATNESS)
        {
            self.edges
                .extend_from_slice(&[p0[0], p0[1], p3[0], p3[1]]);
            return;
        }
        let p01 = midpoint(p0, p1);
        let p12 = midpoint(p1, p2);
        let p23 = midpoint(p2, p3);
        let pp = midpoint(p01, p12);
        let qq = midpoint(p12, p23);
        let mid = midpoint(pp, qq);
        self.cubic(p0, p01, pp, mid, depth + 1);
        self.cubic(mid, qq, p23, p3, depth + 1);
    }
}

impl Decomposer for EdgeBuilder {
    fn move_to(&mut self, to: &Vector) -> TtResult<()> {
        self.current = to_pixels(to);
        Ok(())
    }

    fn line_to(&mut self, to: &Vector) -> TtResult<()> {
        self.push(to_pixels(to));
        Ok(())
    }

    fn conic_to(&mut self, control: &Vector, to: &Vector) -> TtResult<()> {
        self.flatten_quad(to_pixels(control), to_pixels(to));
        Ok(())
    }

    fn cubic_to(&mut self, control1: &Vector, control2: &Vector, to: &Vector) -> TtResult<()> {
        self.flatten_cubic(to_pixels(control1), to_pixels(control2), to_pixels(to));
        Ok(())
    }
}

/// Converts one 26.6 outline point to `f32` pixel coordinates.
#[inline]
fn to_pixels(v: &Vector) -> [f32; 2] {
    [v.x as f32 * COORD_SCALE, v.y as f32 * COORD_SCALE]
}

/// Midpoint of a segment.
#[inline]
fn midpoint(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

/// Shortest distance from `p` to the segment `a..b`, in pixels.
fn point_segment_distance(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let vx = b[0] - a[0];
    let vy = b[1] - a[1];
    let len2 = vx * vx + vy * vy;
    if len2 <= f32::EPSILON {
        let dx = p[0] - a[0];
        let dy = p[1] - a[1];
        return libm::sqrtf(dx * dx + dy * dy);
    }
    let t = (((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / len2).clamp(0.0, 1.0);
    let dx = p[0] - (a[0] + t * vx);
    let dy = p[1] - (a[1] + t * vy);
    libm::sqrtf(dx * dx + dy * dy)
}

/// Renders `source` into `target` on the GPU.
///
/// Returns `true` when the GPU produced the bitmap contents, `false`
/// when the caller must run the CPU scan converter instead (pipeline
/// unavailable, flattening failure, or any driver error — all logged).
/// Only non-zero coverage pixels are written, so untouched target bytes
/// keep their previous contents exactly like the CPU path.
pub fn try_render(source: &Outline, target: &mut Bitmap) -> bool {
    let mut builder = EdgeBuilder {
        edges: Vec::with_capacity(source.n_points as usize * 4),
        current: [0.0; 2],
    };
    if decompose(source, &mut builder).is_err() {
        return false;
    }
    let width = target.width as usize;
    let height = target.rows as usize;
    if width == 0 || height == 0 {
        return false;
    }
    if builder.edges.is_empty() {
        // Nothing to draw: the CPU path produces no spans either.
        return false;
    }
    let even_odd = i32::from(source.flags & OUTLINE_EVEN_ODD_FILL != 0);
    with_pipeline(
        |pipeline| match enqueue(pipeline, &builder.edges, even_odd, width, height, target) {
            Ok(()) => true,
            Err(error) => {
                log_debug!("the HOT rasterizer fell back to the CPU path: {error}");
                false
            }
        },
    )
    .unwrap_or(false)
}

/// Renders `source` into the shared coverage target without a host
/// round-trip, returning the region of the allocation it landed in.
///
/// This is [`try_render`] with its last two steps removed: instead of
/// reading the coverage back into a `Vec` and blitting it into a [`Bitmap`],
/// the kernel writes it straight into the allocation a [`attach_shared_coverage`]
/// imported, and the caller — a renderer whose glyph cache is a `VkImage` —
/// takes it from there as memory its own device owns. The pixels are
/// bottom-up in the same outline space [`blit_coverage`] flips for a bitmap;
/// a GPU caller copies the region as it stands and applies its own
/// orientation.
///
/// Unlike [`try_render`], this overwrites every pixel of the region, the
/// zero-coverage ones included: the whole box belongs to this glyph, so a
/// caller must not share a region between two glyphs.
///
/// Returns `None` — and the caller must then run the CPU scan converter or
/// [`try_render`], as it would for a `false` from that function — when no
/// target is attached, the glyph does not fit the allocation, there are no
/// edges to draw, or any driver step fails. Errors are logged at debug level.
///
/// When this returns `Some`, the kernel has completed and the acquire window
/// has closed, so the coverage is readable by the exporting device. It is a
/// host-visible hand-off, not a signaled one: the calling thread waits for
/// this glyph's kernel instead of queueing work behind a shared semaphore.
pub fn try_render_shared(source: &Outline, width: usize, height: usize) -> Option<CoverageRegion> {
    if width == 0 || height == 0 {
        return None;
    }
    let mut builder = EdgeBuilder {
        edges: Vec::with_capacity(source.n_points as usize * 4),
        current: [0.0; 2],
    };
    if decompose(source, &mut builder).is_err() || builder.edges.is_empty() {
        return None;
    }
    let even_odd = i32::from(source.flags & OUTLINE_EVEN_ODD_FILL != 0);
    with_pipeline(
        |pipeline| match enqueue_shared(pipeline, &builder.edges, even_odd, width, height) {
            Ok(region) => Some(region),
            Err(error) => {
                log_debug!("the HOT shared path fell back to the caller: {error}");
                None
            }
        },
    )
    .flatten()
}
/// Uploads the edges, launches the kernel over the pixel grid, and
/// copies the non-zero coverage pixels into `target`.
///
/// # Errors
///
/// Returns [`codevar_ocl::Error::InvalidArgument`] when the geometry
/// does not fit the kernel's 32-bit parameters, and the underlying
/// driver error for any buffer, argument, enqueue, wait, or read
/// failure.
fn enqueue(
    pipeline: &mut GpuPipeline,
    edges: &[f32],
    even_odd: i32,
    width: usize,
    height: usize,
    target: &mut Bitmap,
) -> Result<(), codevar_ocl::Error> {
    let (n_edges, width_i, height_i) = kernel_params(edges.len() / 4, width, height)?;
    let coverage_len = width
        .checked_mul(height)
        .ok_or(codevar_ocl::Error::InvalidArgument {
            what: "the target dimensions overflow the coverage buffer size",
        })?;
    let GpuPipeline {
        queue,
        kernel,
        coverage: _,
    } = pipeline;
    let context = queue.context();
    let edge_buf = Buffer::from_slice(context, edges)?;
    let coverage_buf = Buffer::new(context, coverage_len)?;
    kernel.set_arg(0, Arg::Buffer(&edge_buf))?;
    kernel.set_arg(1, Arg::I32(n_edges))?;
    kernel.set_arg(2, Arg::I32(width_i))?;
    kernel.set_arg(3, Arg::I32(height_i))?;
    kernel.set_arg(4, Arg::I32(even_odd))?;
    kernel.set_arg(5, Arg::Buffer(&coverage_buf))?;
    let event = queue.enqueue_kernel(kernel, &[width, height], None)?;
    event.wait()?;
    let mut coverage = vec![0u8; coverage_len];
    queue.read_buffer(&coverage_buf, 0, &mut coverage)?;
    blit_coverage(target, &coverage, width, height);
    Ok(())
}

/// Reads the three geometry values `font_raster` takes as `int`s.
///
/// # Errors
///
/// Returns [`codevar_ocl::Error::InvalidArgument`] for any value that does
/// not fit a 32-bit parameter.
#[inline]
fn kernel_params(n_edges: usize, width: usize, height: usize) -> Result<(i32, i32, i32), codevar_ocl::Error> {
    let checked =
        |value: usize, what| i32::try_from(value).map_err(|_| codevar_ocl::Error::InvalidArgument { what });
    Ok((
        checked(n_edges, "the edge count exceeds the kernel's 32-bit parameter")?,
        checked(width, "the target width exceeds the kernel's 32-bit parameter")?,
        checked(height, "the target height exceeds the kernel's 32-bit parameter")?,
    ))
}

/// Launches `font_raster` over the shared allocation inside one acquire
/// window, and waits for the render.
///
/// The window is the whole reason this path is safe: the allocation belongs
/// to the exporting device, and OpenCL may only write it between the acquire
/// and the release that [`SharedBuffer::acquire`] and its `Drop` put around
/// the kernel. Waiting on the kernel event before the guard is dropped is
/// what keeps the release behind the write in the driver's own order.
///
/// # Errors
///
/// Returns the [`codevar_ocl::Error`] of any failed buffer, argument,
/// enqueue, wait or window command, widened into
/// [`codevar_ocl_interop::Error`], and `InvalidArgument` when no target is
/// attached, the glyph is larger than the allocation, or a parameter does
/// not fit 32 bits.
fn enqueue_shared(
    pipeline: &mut GpuPipeline,
    edges: &[f32],
    even_odd: i32,
    width: usize,
    height: usize,
) -> Result<CoverageRegion, InteropError> {
    let coverage_len = width
        .checked_mul(height)
        .ok_or(codevar_ocl::Error::InvalidArgument {
            what: "the target dimensions overflow the shared coverage size",
        })?;
    let GpuPipeline {
        queue,
        kernel,
        coverage,
    } = pipeline;
    let shared = coverage
        .as_ref()
        .ok_or(codevar_ocl::Error::InvalidArgument {
            what: "no shared coverage allocation is attached to the HOT pipeline",
        })?;
    if shared.buffer().size() < coverage_len {
        return Err(codevar_ocl::Error::InvalidArgument {
            what: "the glyph is larger than the attached shared allocation",
        }
        .into());
    }
    let (n_edges, width_i, height_i) = kernel_params(edges.len() / 4, width, height)?;
    let edge_buf = Buffer::from_slice(queue.context(), edges)?;
    let _share = shared.acquire(queue)?;
    kernel.set_arg(0, Arg::Buffer(&edge_buf))?;
    kernel.set_arg(1, Arg::I32(n_edges))?;
    kernel.set_arg(2, Arg::I32(width_i))?;
    kernel.set_arg(3, Arg::I32(height_i))?;
    kernel.set_arg(4, Arg::I32(even_odd))?;
    kernel.set_arg(5, Arg::Buffer(shared.buffer()))?;
    let event = queue.enqueue_kernel(kernel, &[width, height], None)?;
    event.wait()?;
    drop(_share);
    Ok(CoverageRegion { width, height })
}

/// Copies non-zero coverage pixels into the bitmap, translating the
/// kernel's bottom-up outline space into memory rows the same way
/// [`codevar_truetype_core::Span`] emission does (logical scanline `y`
/// lives at memory row `rows - 1 - y` for a positive pitch).
fn blit_coverage(target: &mut Bitmap, coverage: &[u8], width: usize, height: usize) {
    for logical_y in 0..height {
        let start = logical_y * width;
        let Some(src) = coverage.get(start..start + width) else {
            return;
        };
        if src.iter().all(|&value| value == 0) {
            continue;
        }
        let index = (height - 1 - logical_y) as u32;
        let Some(row) = target.row_mut(index) else {
            continue;
        };
        if row.len() < width {
            continue;
        }
        for (x, &value) in src.iter().enumerate() {
            if value != 0 {
                row[x] = value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grays::GrayRaster;
    use codevar_truetype_core::{BBox, PixelMode, Raster, RasterFlags, RasterParams};

    /// One outline point in 26.6 units (64 = one pixel).
    fn pt(x: i64, y: i64) -> Vector {
        Vector { x, y }
    }

    /// A counter-clockwise square in 26.6 units.
    fn square(flags: i32) -> Outline {
        Outline {
            n_contours: 1,
            n_points: 4,
            points: vec![pt(0, 0), pt(64, 0), pt(64, 64), pt(0, 64)],
            tags: vec![1, 1, 1, 1],
            contours: vec![3],
            flags,
        }
    }

    /// Two identical overlapping squares: filled under the non-zero
    /// rule, a hole under the even-odd rule.
    fn double_square(flags: i32) -> Outline {
        Outline {
            n_contours: 2,
            n_points: 8,
            points: vec![
                pt(0, 0),
                pt(64, 0),
                pt(64, 64),
                pt(0, 64),
                pt(0, 0),
                pt(64, 0),
                pt(64, 64),
                pt(0, 64),
            ],
            tags: vec![1, 1, 1, 1, 1, 1, 1, 1],
            contours: vec![3, 7],
            flags,
        }
    }

    fn gray_target(rows: u32, width: u32) -> Bitmap {
        Bitmap::new_sized(rows, width, PixelMode::Gray, 256).unwrap()
    }

    fn render(outline: &Outline, flags: RasterFlags, target: &mut Bitmap) -> TtResult<()> {
        let mut raster = GrayRaster::new();
        let mut params = RasterParams {
            target,
            source: outline,
            flags,
            gray_spans: None,
            user: None,
            clip_box: BBox::new(),
        };
        raster.render(&mut params)
    }

    /// GPU and CPU cover the same axis-aligned geometry exactly at
    /// quarter-pixel boundaries; elsewhere the 2x2 kernel grid may
    /// differ from the exact-area CPU scan converter by at most one
    /// quarter of the 0..255 range.
    fn assert_similar(cpu: &Bitmap, gpu: &Bitmap, max_diff: u8) {
        assert_eq!(cpu.buffer.len(), gpu.buffer.len());
        for (index, (&cpu_value, &gpu_value)) in cpu.buffer.iter().zip(&gpu.buffer).enumerate() {
            let diff = cpu_value.abs_diff(gpu_value);
            assert!(
                diff <= max_diff,
                "pixel {index}: cpu={cpu_value} gpu={gpu_value} (diff {diff} > {max_diff})"
            );
        }
    }

    #[test]
    fn hot_flag_renders_full_pixel_square() {
        let outline = square(0);
        let mut target = gray_target(1, 1);
        render(&outline, RasterFlags::AA | RasterFlags::HOT, &mut target).unwrap();
        assert_eq!(target.buffer[0], 255);
    }

    #[test]
    fn hot_matches_cpu_on_half_pixel_square() {
        // Half a pixel wide: both paths must produce exactly half
        // coverage (128).
        let outline = Outline {
            n_contours: 1,
            n_points: 4,
            points: vec![pt(0, 0), pt(32, 0), pt(32, 64), pt(0, 64)],
            tags: vec![1, 1, 1, 1],
            contours: vec![3],
            flags: 0,
        };
        let mut cpu = gray_target(1, 1);
        let mut gpu = gray_target(1, 1);
        render(&outline, RasterFlags::AA, &mut cpu).unwrap();
        render(&outline, RasterFlags::AA | RasterFlags::HOT, &mut gpu).unwrap();
        assert_eq!(cpu.buffer[0], 128);
        assert_eq!(gpu.buffer, cpu.buffer);
    }

    #[test]
    fn hot_matches_cpu_on_larger_square() {
        // A 4x4 pixel square inset by half a pixel on every side:
        // full-coverage interior, half-coverage borders, quarter
        // corners. With no GPU the render falls back to the CPU and the
        // buffers match trivially.
        let outline = Outline {
            n_contours: 1,
            n_points: 4,
            points: vec![
                pt(32, 32),
                pt(4 * 64 - 32, 32),
                pt(4 * 64 - 32, 4 * 64 - 32),
                pt(32, 4 * 64 - 32),
            ],
            tags: vec![1, 1, 1, 1],
            contours: vec![3],
            flags: 0,
        };
        let mut cpu = gray_target(4, 4);
        let mut gpu = gray_target(4, 4);
        render(&outline, RasterFlags::AA, &mut cpu).unwrap();
        render(&outline, RasterFlags::AA | RasterFlags::HOT, &mut gpu).unwrap();
        assert!(cpu.buffer.contains(&255), "interior must be fully covered");
        assert_similar(&cpu, &gpu, 32);
    }

    #[test]
    fn hot_even_odd_and_nonzero_fill_rules_match_cpu() {
        // Two coincident squares: non-zero rule fills the pixel.
        let mut cpu = gray_target(1, 1);
        let mut gpu = gray_target(1, 1);
        render(&double_square(0), RasterFlags::AA, &mut cpu).unwrap();
        render(&double_square(0), RasterFlags::AA | RasterFlags::HOT, &mut gpu).unwrap();
        assert_eq!(cpu.buffer[0], 255);
        assert_eq!(gpu.buffer[0], 255);

        // The even-odd rule punches the overlap back out. Fresh
        // targets: both rasterizers leave uncovered pixels untouched
        // rather than clearing them.
        let mut cpu = gray_target(1, 1);
        let mut gpu = gray_target(1, 1);
        render(&double_square(OUTLINE_EVEN_ODD_FILL), RasterFlags::AA, &mut cpu).unwrap();
        render(
            &double_square(OUTLINE_EVEN_ODD_FILL),
            RasterFlags::AA | RasterFlags::HOT,
            &mut gpu,
        )
        .unwrap();
        assert_eq!(cpu.buffer[0], 0);
        assert_eq!(gpu.buffer[0], 0);
    }

    #[test]
    fn hot_conic_outline_renders_full_pixel() {
        // A collinear conic control point must flatten to the same
        // straight edge (this exercises the flattener on both paths).
        let outline = Outline {
            n_contours: 1,
            n_points: 5,
            points: vec![pt(0, 0), pt(16, 0), pt(64, 0), pt(64, 64), pt(0, 64)],
            tags: vec![1, 0, 1, 1, 1],
            contours: vec![4],
            flags: 0,
        };
        let mut target = gray_target(1, 1);
        render(&outline, RasterFlags::AA | RasterFlags::HOT, &mut target).unwrap();
        assert_eq!(target.buffer[0], 255);
    }
}
