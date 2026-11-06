// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and
// limitations under the License.

//! Headless pixel probe of the text pipe's glyph pages.
//!
//! One synthetic glyph is drawn three ways and the presented pixels are
//! compared, so a device page has to agree with the host page it replaced:
//!
//! * `host` — the atlas mirrors its coverage and the pipe packs it into the
//!   staging buffer (the reference).
//! * `overlay` — a device-backed atlas, whose coverage arrives as a host
//!   rasterization and is memcpy'd into the page's storage.
//! * `dmabuf` — a device-backed atlas, whose coverage is written through the
//!   dma-buf that `TextPipe::page_dmabuf` exports, at the addresses the
//!   reservation names. This is the path a GPU rasterizer takes.
//!
//! The glyph sits at odd coordinates and the coverage is a gradient, so a wrong
//! row pitch, a swapped origin or an off-by-one in the copy shifts pixels
//! instead of moving a symmetric shape onto itself.
//!
//! ```text
//! cargo run -p codevar-gpu-vk --example probe_text_pages
//! ```

use std::error::Error;

use codevar_gpu_vk::base::{Compositor, block_on};
use codevar_gpu_vk::pipeline::{OwnedFd, PipelineContext};
use codevar_kuitui_text::{
    AtlasConfig, DeviceSlot, GlyphKey, GlyphQuad, Reservation, TextAtlas, TextFrame, TextPipe,
};

/// Width of the presented image.
const WIDTH: u32 = 96;
/// Height of the presented image.
const HEIGHT: u32 = 96;
/// Side of one atlas page.
const PAGE_SIZE: u32 = 64;
/// Width of the probe glyph.
const GLYPH_WIDTH: u32 = 13;
/// Height of the probe glyph.
const GLYPH_HEIGHT: u32 = 9;
/// Where the glyph is drawn, chosen odd so a pitch error cannot hide.
const POSITION: [f32; 2] = [7.0, 11.0];
/// The glyph color.
const COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
/// The glyph key.
const KEY: GlyphKey = GlyphKey::from_raw(1);
/// `DRM_FORMAT_MOD_LINEAR`, which is what makes the readback a plain raster.
const LINEAR_MODIFIER: u64 = 0;

/// Every failure is boxed so `?` covers Vulkan, compositor and OS errors.
type ProbeResult<T> = Result<T, Box<dyn Error>>;

/// How the probe glyph's coverage reaches its page.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// A host atlas: the atlas mirrors the coverage and the pipe stages it
    /// through the packing buffer. This is the reference.
    Host,
    /// A device atlas whose coverage arrives from the CPU, so the pipe
    /// memcpy's it into the page's own storage.
    Overlay,
    /// A device atlas whose coverage is written at the addresses the
    /// reservation names, through the page's dma-buf. This is the path a GPU
    /// rasterizer takes.
    Dmabuf,
}

/// The coverage of the probe glyph: a ramp that is never symmetric.
fn coverage() -> Vec<u8> {
    (0..GLYPH_WIDTH * GLYPH_HEIGHT)
        .map(|i| (i / GLYPH_WIDTH * 9 + i % GLYPH_WIDTH * 23) as u8)
        .collect()
}

/// Adds the probe glyph and its quad to the frame, returning the device slot
/// when the caller has to write the coverage itself.
fn record(frame: &TextFrame, mode: Mode) -> ProbeResult<Option<DeviceSlot>> {
    let bitmap = coverage();
    frame.begin_frame();
    let shaped = frame.with(|atlas, mesh| -> ProbeResult<Option<DeviceSlot>> {
        let reserved = match mode {
            Mode::Host | Mode::Overlay => atlas
                .insert(KEY, GLYPH_WIDTH, GLYPH_HEIGHT, &bitmap)
                .map_err(|err| -> Box<dyn Error> { err.to_string().into() })?
                .map(|glyph| Reservation {
                    glyph,
                    coverage: None,
                }),
            Mode::Dmabuf => atlas
                .reserve(KEY, GLYPH_WIDTH, GLYPH_HEIGHT)
                .map_err(|err| -> Box<dyn Error> { err.to_string().into() })?,
        };
        let reserved = reserved.ok_or("the glyph produced no reservation")?;
        mesh.push(&GlyphQuad::from_slot(reserved.glyph, POSITION, COLOR, None));
        Ok(reserved.coverage)
    });
    let slot = shaped?;
    frame.mark_dirty();
    Ok(slot)
}

/// Draws the probe glyph in `mode` and returns the presented pixels plus the
/// number of them that are not the cleared background.
fn render(mode: Mode) -> ProbeResult<(Vec<u8>, usize)> {
    let device_backed = mode != Mode::Host;
    let context = PipelineContext::new(WIDTH, HEIGHT, &[LINEAR_MODIFIER])?;
    let atlas = TextAtlas::new(AtlasConfig {
        page_size: PAGE_SIZE,
        max_pages: 1,
        device_backed,
    })?;
    let frame = TextFrame::new(atlas)?;
    let mut pipe = TextPipe::new(&context, frame.clone())?;
    let slot = record(&frame, mode)?;

    // The dma-buf run is the stand-in for a GPU rasterizer: it fills the page
    // through the descriptor the pipe exported, before the frame is submitted.
    if mode == Mode::Dmabuf {
        let fd = pipe
            .page_dmabuf(0)?
            .ok_or("a device-backed page has no dma-buf")?;
        let slot = slot.ok_or("the reservation has no device slot")?;
        write_through_dmabuf(&fd, &slot)?;
    }

    let mut compositor = Compositor::new(&context);
    compositor.start()?;
    compositor.add_pipe(pipe)?;
    let _frame_one = block_on(compositor.composite_frame(None))?;
    let _frame_two = block_on(compositor.composite_frame(None))?;
    // SAFETY: the device is the one this context created, and waiting it idle
    // is what makes the presented bytes readable by the host afterwards.
    let idle = unsafe { context.device().device_wait_idle() };
    idle.map_err(|err| -> Box<dyn Error> { format!("device wait idle failed: {err:?}").into() })?;

    let pixels = present_pixels(&context)?;
    let painted = pixels
        .chunks_exact(4)
        .filter(|pixel| pixel[0] | pixel[1] | pixel[2] != 0)
        .count();
    Ok((pixels, painted))
}

/// Writes the probe glyph into the page storage through its dma-buf, at the
/// addresses the reservation named.
fn write_through_dmabuf(fd: &OwnedFd, slot: &DeviceSlot) -> ProbeResult<()> {
    let bitmap = coverage();
    let bytes = usize::try_from(PAGE_SIZE * PAGE_SIZE)?;
    // SAFETY: the descriptor names a `page_size * page_size` allocation the
    // pipe keeps alive for as long as this mapping is used, and the mapping is
    // released before returning on every path below.
    let addr = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd.as_raw(),
            0,
        )
    };
    if addr == libc::MAP_FAILED {
        return Err("mmap of the page dma-buf failed".into());
    }
    let mapped = addr.cast::<u8>();
    for row in 0..slot.height {
        let source = &bitmap[row * slot.width..(row + 1) * slot.width];
        let address = slot
            .row_address(row)
            .ok_or("the slot row is outside the page")?;
        // SAFETY: `address..address + width` is inside the slot, which the
        // atlas placed inside the page, and no GPU work is in flight.
        unsafe { core::ptr::copy_nonoverlapping(source.as_ptr(), mapped.add(address), source.len()) };
    }
    // SAFETY: `addr` is the mapping made above and `bytes` its length.
    unsafe { libc::munmap(addr, bytes) };
    Ok(())
}

/// Reads the presented image out of the render target's dma-buf.
fn present_pixels(context: &PipelineContext) -> ProbeResult<Vec<u8>> {
    let target = context.present_target();
    if target.modifier != LINEAR_MODIFIER {
        return Err(format!("the render target is not linear: {:#x}", target.modifier).into());
    }
    let stride = usize::try_from(target.stride)?;
    let row_bytes = stride.saturating_mul(usize::try_from(target.height)?);
    let length = usize::try_from(target.offset)?.saturating_add(row_bytes);
    // SAFETY: the descriptor names the render target allocation, which the
    // context owns and outlives this mapping.
    let addr = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            length,
            libc::PROT_READ,
            libc::MAP_SHARED,
            target.dmabuf_fd.as_raw(),
            0,
        )
    };
    if addr == libc::MAP_FAILED {
        return Err("mmap of the present dma-buf failed".into());
    }
    let mut pixels = vec![0u8; row_bytes];
    // SAFETY: the plane starts at `offset` and is `stride * height` bytes,
    // which is the window mapped above.
    unsafe {
        core::ptr::copy_nonoverlapping(
            (addr as *const u8).add(usize::try_from(target.offset)?),
            pixels.as_mut_ptr(),
            row_bytes,
        );
    }
    // SAFETY: `addr` is the mapping made above and `length` its size.
    unsafe { libc::munmap(addr, length) };
    Ok(pixels)
}

/// The first byte offset at which `left` and `right` disagree.
fn first_difference(left: &[u8], right: &[u8]) -> Option<usize> {
    left.iter().zip(right).position(|(a, b)| a != b)
}

fn main() {
    match run() {
        Ok(()) => println!("probe_text_pages: ok"),
        Err(error) => {
            eprintln!("probe_text_pages: {error}");
            std::process::exit(1);
        }
    }
}

fn run() -> ProbeResult<()> {
    let modes = [
        ("host", Mode::Host),
        ("overlay", Mode::Overlay),
        ("dmabuf", Mode::Dmabuf),
    ];
    let (host, host_painted) = render(Mode::Host)?;
    let mut painted = vec![host_painted];

    for (name, mode) in modes.iter().skip(1) {
        let (pixels, count) = render(*mode)?;
        if let Some(offset) = first_difference(&host, &pixels) {
            return Err(format!(
                "{name} differs from host at byte {offset} (row {}, column {})",
                offset / (host.len() / usize::try_from(HEIGHT)?),
                offset % (host.len() / usize::try_from(HEIGHT)?),
            )
            .into());
        }
        painted.push(count);
    }

    for ((name, _mode), count) in modes.iter().zip(&painted) {
        if *count == 0 {
            return Err(format!("{name} drew nothing").into());
        }
        println!("{name}: {count} painted pixels");
    }
    if painted.iter().any(|count| *count != host_painted) {
        return Err("the paths painted a different number of pixels".into());
    }
    Ok(())
}
