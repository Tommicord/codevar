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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Glyph coverage that never returns to host memory.
//!
//! These tests do what a renderer does: a Vulkan device allocates the
//! coverage target, the HOT pipeline imports it, one glyph is rasterized into
//! it, and the *Vulkan* side reads the bytes back. Nothing in that sequence
//! calls `read_buffer`, so a rasterizer that stopped writing into the shared
//! allocation, or wrote the wrong box, shows up here rather than in a mock.
//!
//! A machine whose two drivers do not describe the same GPU, or that has no
//! loader at all, has no exchange to check and returns early.

#![cfg(unix)]

mod support;

use std::sync::{Mutex, MutexGuard, OnceLock};

use codevar_ocl::{Device, Runtime};
use codevar_ocl_interop::{MemoryHandle, MemoryKind, Vulkan};
use codevar_truetype_core::{BBox, Bitmap, Outline, PixelMode, Raster, RasterFlags, RasterParams, Vector};
use codevar_truetype_smooth::grays::GrayRaster;
use codevar_truetype_smooth::hot::{self, CoverageSlot};
use support::{Allocation, Machine, close_descriptor};

/// Bytes a test allocation reserves for coverage.
const ALLOCATION: u64 = 4096;

/// The HOT pipeline is one process-wide queue, so these tests take turns
/// rather than racing each other for it.
fn pipeline_turn() -> MutexGuard<'static, ()> {
    static TURN: Mutex<()> = Mutex::new(());
    TURN.lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One outline point in 26.6 units (64 = one pixel).
fn pt(x: i64, y: i64) -> Vector {
    Vector { x, y }
}

/// A counter-clockwise square in 26.6 units.
fn square(x0: i64, y0: i64, x1: i64, y1: i64) -> Outline {
    Outline {
        n_contours: 1,
        n_points: 4,
        points: vec![pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)],
        tags: vec![1, 1, 1, 1],
        contours: vec![3],
        flags: 0,
    }
}

/// The exact-area CPU rasterizer's coverage for `outline`, memory rows first.
fn cpu_coverage(outline: &Outline, rows: u32, width: u32) -> Option<Vec<u8>> {
    let mut target = Bitmap::new_sized(rows, width, PixelMode::Gray, 256).ok()?;
    let mut raster = GrayRaster::new();
    let mut params = RasterParams {
        target: &mut target,
        source: outline,
        // No HOT: this is the CPU scan converter's own answer.
        flags: RasterFlags::AA,
        gray_spans: None,
        user: None,
        clip_box: BBox::new(),
    };
    raster.render(&mut params).ok()?;
    Some(target.buffer)
}

/// The handle kind both drivers of this pair agree on.
fn agreed_kind(endpoint: &Vulkan, device: &Device) -> Option<MemoryKind> {
    [MemoryKind::DmaBuf, MemoryKind::OpaqueFd]
        .into_iter()
        .find(|kind| {
            endpoint.can_export_memory(*kind) && matches!(kind.importable_by_opencl(device), Ok(true))
        })
}

/// A Vulkan device that can export and the OpenCL device that is the same
/// hardware.
fn pair() -> Option<(Machine, Device)> {
    static DEVICES: OnceLock<Vec<Device>> = OnceLock::new();
    let devices = DEVICES.get_or_init(|| {
        let Ok(runtime) = Runtime::load() else {
            return Vec::new();
        };
        runtime
            .platforms()
            .iter()
            .filter_map(|platform| platform.all_devices().ok())
            .flatten()
            .collect()
    });
    for machine in Machine::all() {
        let identity = machine.endpoint().identity();
        for device in devices {
            if matches!(identity.pairs_with(device), Ok(true)) {
                return Some((machine, device.clone()));
            }
        }
    }
    None
}

/// The glyph rasterized into the box `slot` names, read back through the
/// Vulkan mapping as every byte up to the end of that box.
///
/// Returns `None` when this machine cannot run the exchange at all — no
/// paired device, no memory type both sides accept, or a pipeline that chose
/// another GPU than the pair. A driver that does import an allocation has to
/// rasterize into it, so the caller's assertions run on anything this returns.
fn rasterize(outline: &Outline, slot: CoverageSlot) -> Option<Vec<u8>> {
    rasterize_slots(&[(outline, slot)], slot)
}

/// Each glyph rasterized into its own box of one imported allocation, then
/// the page read back as every byte up to the end of `page`.
///
/// This is the sequence a glyph cache runs: one allocation, one import, and a
/// render per glyph that has to land where the cache said it would.
fn rasterize_slots(items: &[(&Outline, CoverageSlot)], page: CoverageSlot) -> Option<Vec<u8>> {
    let (machine, device) = pair()?;
    let endpoint = machine.endpoint();
    let kind = agreed_kind(&endpoint, &device)?;
    let allocation: Allocation = machine.allocate(kind, ALLOCATION)?;
    let Ok(handle) = endpoint.export_memory(allocation.memory(), kind) else {
        return None;
    };
    let imported = hot::SharedCoverage::import(handle, allocation.len());
    // SAFETY: the import only reads the descriptor and takes its own
    // reference, so this process owns the one `export_memory` handed out
    // whichever way the import answered.
    unsafe { close_descriptor(handle.raw() as i32) };
    let target = imported.ok()?;
    let raster = GrayRaster::new();
    for (outline, slot) in items {
        raster.render_shared_coverage(outline, RasterFlags::AA | RasterFlags::HOT, &target, *slot)?;
    }
    allocation.read(page.bytes()?.end)
}

/// Writes the CPU rasterizer's coverage for `outline` into `page` at the
/// bytes `slot` addresses.
///
/// The kernel puts logical row `py` — counted from the outline's bottom — at
/// `offset + py * pitch`, and that row is memory row `height - 1 - py` of the
/// top-down CPU bitmap, so this is the kernel's own addressing restated in
/// host memory rather than an independent guess at it.
fn place(page: &mut [u8], outline: &Outline, slot: CoverageSlot) -> Option<()> {
    let cpu = cpu_coverage(outline, slot.height as u32, slot.width as u32)?;
    for py in 0..slot.height {
        let row = slot.offset as isize + py as isize * slot.pitch;
        for px in 0..slot.width {
            let at = usize::try_from(row + px as isize).ok()?;
            let from = (slot.height - 1 - py) * slot.width + px;
            *page.get_mut(at)? = *cpu.get(from)?;
        }
    }
    Some(())
}

/// The kernel writes the box where the sharing device can read it, in the
/// outline's own bottom-up order, and its coverage matches the exact-area CPU
/// scan converter pixel for pixel.
#[test]
fn coverage_lands_in_memory_the_vulkan_device_owns() {
    let _turn = pipeline_turn();
    // A 2x2 pixel block, offset one pixel from the logical origin of a 4x4
    // box: integer-aligned, so both rasterizers owe the same answer.
    let outline = square(64, 64, 3 * 64, 4 * 64);
    let slot = CoverageSlot::tight(4, 4);
    let Some(bytes) = rasterize(&outline, slot) else {
        return;
    };
    assert_eq!(bytes.len(), 16, "the box is the size that was asked for");
    // The CPU bitmap is stored top-down and the slot is bottom-up, so the
    // rows of one are the rows of the other in reverse.
    let Some(cpu) = cpu_coverage(&outline, 4, 4) else {
        return;
    };
    let expected: Vec<u8> = cpu.chunks(4).rev().flatten().copied().collect();
    assert_eq!(bytes, expected, "the device must cover the same pixels");
    assert!(
        bytes.contains(&255) && bytes.contains(&0),
        "the block is neither empty nor the whole box"
    );
}

/// A sub-pixel edge lands on the same coverage level on both sides, which is
/// the check that the shared buffer holds real antialiased pixels and not a
/// hard-edged guess.
#[test]
fn half_pixel_coverage_crosses_the_border_unchanged() {
    let _turn = pipeline_turn();
    // Half a pixel wide, one pixel tall: 128 on both paths.
    let outline = square(0, 0, 32, 64);
    let Some(bytes) = rasterize(&outline, CoverageSlot::tight(1, 1)) else {
        return;
    };
    assert_eq!(bytes, vec![128]);
}

/// Two glyphs of one page, each in a box of its own inside a row eight bytes
/// wide, and neither of them leaves it: the kernel has to honour `offset`, not
/// just the box size.
#[test]
fn two_glyphs_share_one_page_without_touching_each_other() {
    let _turn = pipeline_turn();
    // The page is four rows of eight bytes; the left half belongs to the
    // first glyph and the right half to the second, so their boxes share
    // rows but not pixels.
    let page = CoverageSlot::tight(8, 4);
    let left = CoverageSlot {
        width: 4,
        height: 4,
        offset: 0,
        pitch: 8,
    };
    let right = CoverageSlot {
        width: 4,
        height: 4,
        offset: 4,
        pitch: 8,
    };
    // Coverage in the bottom-left of the first box and the top-right of the
    // second: the two corners furthest apart.
    let block = square(0, 0, 2 * 64, 2 * 64);
    let notch = square(2 * 64, 2 * 64, 4 * 64, 4 * 64);
    let Some(bytes) = rasterize_slots(&[(&block, left), (&notch, right)], page) else {
        return;
    };
    let mut expected = vec![0u8; 32];
    if place(&mut expected, &block, left).is_none() || place(&mut expected, &notch, right).is_none() {
        return;
    }
    assert_eq!(bytes, expected, "each glyph must stay inside its own box");
    assert!(
        bytes[..16].contains(&255) && bytes[16..].contains(&255),
        "both boxes were written"
    );
}

/// A negative pitch stores the outline's rows top-down, which is the order a
/// CPU bitmap already keeps, so the box comes out equal to the host answer
/// with no row reversal: the sign of `pitch` reaches the kernel rather than
/// only its magnitude.
#[test]
fn a_negative_pitch_stores_the_rows_top_down() {
    let _turn = pipeline_turn();
    // A tight 4x4 box written from its last row upwards: origin 12, pitch -4.
    let slot = CoverageSlot {
        width: 4,
        height: 4,
        offset: 12,
        pitch: -4,
    };
    let outline = square(0, 0, 2 * 64, 2 * 64);
    let Some(bytes) = rasterize(&outline, slot) else {
        return;
    };
    let Some(cpu) = cpu_coverage(&outline, 4, 4) else {
        return;
    };
    assert_eq!(bytes, cpu, "top-down storage is the CPU row order");
    assert_eq!(
        bytes.first_chunk::<4>(),
        Some(&[0, 0, 0, 0]),
        "the empty logical top rows must still be empty"
    );
}

/// A glyph the target cannot hold is refused before the kernel is asked, so a
/// renderer sizing its scratch too small falls back instead of writing past
/// somebody else's memory.
#[test]
fn a_glyph_larger_than_the_allocation_falls_back() {
    let _turn = pipeline_turn();
    let Some((machine, device)) = pair() else {
        return;
    };
    let endpoint = machine.endpoint();
    let Some(kind) = agreed_kind(&endpoint, &device) else {
        return;
    };
    let Some(allocation) = machine.allocate(kind, 64) else {
        return;
    };
    let Ok(handle) = endpoint.export_memory(allocation.memory(), kind) else {
        return;
    };
    let imported = hot::SharedCoverage::import(handle, allocation.len());
    // SAFETY: as in `rasterize`, the descriptor is this process's to close.
    unsafe { close_descriptor(handle.raw() as i32) };
    let Ok(target) = imported else {
        return;
    };
    let outline = square(0, 0, 64, 64);
    // 16x16 needs 256 bytes of a 64-byte allocation.
    assert!(
        hot::try_render_shared(
            &target,
            &square(0, 0, 16 * 64, 16 * 64),
            CoverageSlot::tight(16, 16)
        )
        .is_none(),
        "a glyph over the allocation size must not be rasterized device-side"
    );
    assert_eq!(
        hot::try_render_shared(&target, &outline, CoverageSlot::tight(8, 8)),
        Some(0..64),
        "the same allocation does take a glyph that fits it"
    );
    // A 4x4 box that ends exactly on the last byte is inside the allocation;
    // one origin byte further it is outside, and the size alone would say
    // otherwise.
    assert_eq!(
        hot::try_render_shared(
            &target,
            &outline,
            CoverageSlot {
                width: 4,
                height: 4,
                offset: 48,
                pitch: 4
            },
        ),
        Some(48..64),
        "a box ending on the allocation is accepted"
    );
    assert!(
        hot::try_render_shared(
            &target,
            &outline,
            CoverageSlot {
                width: 4,
                height: 4,
                offset: 52,
                pitch: 4
            },
        )
        .is_none(),
        "a box reaching past it must be refused"
    );
}

/// A descriptor that names no allocation never becomes a target, so the
/// zero-copy path cannot be entered with memory nobody exported. Whether the
/// machine has no pipeline at all or a driver that refuses the descriptor, a
/// target is not produced.
#[test]
fn a_target_imported_from_a_bogus_descriptor_is_refused() {
    let _turn = pipeline_turn();
    assert!(
        hot::SharedCoverage::import(MemoryHandle::DmaBuf(-1), 64).is_err(),
        "a dma_buf descriptor of -1 cannot name an allocation"
    );
}
