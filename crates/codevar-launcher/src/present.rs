//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! GPU presentation of the launcher shell.
//!
//! The window backend owns the surface, the compositor owns the GPU:
//! this module ties them together. It builds a [`PipelineContext`] for
//! the window's size and the modifiers the compositor accepts,
//! registers a [`UiPipe`] that replays one [`DrawList`] per frame, and
//! presents the exported dma-buf through [`Window::present`].
//!
//! # Pacing
//!
//! A frame is paced by the compositor instead of a timer:
//! [`Window::request_frame`] asks for a refresh callback *before* the
//! buffer is committed, the callback wakes [`Window::poll`], and only
//! then is the next frame recorded. [`Window::buffer_free`] gates the
//! single present slot, so a frame is never recorded into an image the
//! compositor may still be reading.
//!
//! # Resolution
//!
//! A [`PipelineContext`] exports one image of a fixed size, so a
//! resize rebuilds the whole pipeline. The shell is static, which keeps
//! that cost to the moment the user actually resizes the window.

use crate::driver::Exit;
use crate::wnd::{Window, WndBackendId, WndError, WndEvent, WndImage, WndSize};
use alloc::format;
use codevar_gpu_vk::base::{Compositor, block_on};
use codevar_gpu_vk::pipeline::{OwnedFd, PipelineContext, RenderTarget};
use codevar_gpu_vk::ui::{UiFrame, UiPipe};
use codevar_kuitui::paint::{CornerRadii, DrawList, Paint};
use codevar_kuitui::{Color, Offset, Rect};
use codevar_logger::{log_debug, log_error, log_info, log_warn};
use codevar_time_core::TimeDuration;
use core::ffi::c_int;

/// The present slot the launcher renders into.
const SLOT: u32 = 0;
/// How long one poll waits for the compositor to release the slot.
const RELEASE_POLL: TimeDuration = TimeDuration::from_millis(4);
/// Slice of the GPU wait spent polling window events.
const GPU_POLL_SLICE_MS: c_int = 4;
/// Total wait before a frame's `sync_file` is considered stalled.
const GPU_SYNC_TIMEOUT_MS: c_int = 500;

/// Clear color of the offscreen target, matching the backdrop.
const CLEAR: [f32; 4] = [0.09, 0.08, 0.09, 1.0];
/// The surface behind the shell.
const BACKDROP: Color = Color::new(0.09, 0.08, 0.09, 1.0);
/// The rounded edge of the search field.
const FIELD: Color = Color::new(0.16, 0.15, 0.16, 1.0);
/// The inset surface of the search field.
const FIELD_INNER: Color = Color::new(0.20, 0.19, 0.20, 1.0);
/// Fraction of the window width the search field takes.
const FIELD_WIDTH_RATIO: f32 = 0.55;
/// Narrowest the search field gets.
const MIN_FIELD_WIDTH: f32 = 320.0;
/// Widest the search field gets.
const MAX_FIELD_WIDTH: f32 = 720.0;
/// Height of the search field.
const FIELD_HEIGHT: f32 = 56.0;
/// Corner radius of the search field.
const FIELD_RADIUS: f32 = 28.0;
/// Width of the inset border of the search field.
const FIELD_INSET: f32 = 2.0;
/// Gap kept between the field and the window edges.
const EDGE: f32 = 16.0;

/// Presents the launcher shell on `window`.
///
/// Returns `None` when the window cannot present dma-buf images or when
/// the GPU pipeline could not be built at all; the caller then keeps
/// the window's own shared-memory output. Every other outcome closes
/// the window and is reported as an [`Exit`].
#[must_use]
pub fn run(window: &mut dyn Window) -> Option<Exit> {
    let modifiers = window.dmabuf_modifiers().to_vec();
    if modifiers.is_empty() {
        log_info!("the window offers no dma-buf layouts; keeping the shared-memory surface");
        return None;
    }
    let mut started = false;
    loop {
        match session(window, &modifiers, &mut started) {
            Ok(SessionEnd::Closed) => {
                window.close();
                return Some(Exit::Success);
            }
            Ok(SessionEnd::Resized) => log_debug!("window resized; rebuilding the render target"),
            Err(error) if !started => {
                log_warn!(
                    "the GPU presentation path is unavailable: {}; keeping the shared-memory window",
                    error
                );
                return None;
            }
            Err(error) => {
                log_error!("the render loop failed: {}", error);
                window.close();
                return Some(Exit::Failure);
            }
        }
    }
}

/// Why one presentation session stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionEnd {
    /// The window was closed.
    Closed,
    /// The window changed size, so the fixed-size target must be rebuilt.
    Resized,
}

/// Builds the pipeline for the window's current size and presents until
/// the window closes or resizes.
///
/// `started` is set once the GPU pipeline is live: a failure before
/// that point lets [`run`] fall back to the shared-memory window, while
/// a failure after it ends the session.
fn session(window: &mut dyn Window, modifiers: &[u64], started: &mut bool) -> Result<SessionEnd, WndError> {
    let size = window.size();
    let pipeline = create_pipeline(window.backend(), size, modifiers)?;
    let image = present_image(pipeline.present_target());
    let ui_frame = UiFrame::new();
    let mut compositor = Compositor::new(&pipeline);
    compositor
        .start()
        .map_err(|error| frame_error(window, format!("compositor start failed: {error}")))?;
    let mut ui_pipe = UiPipe::new(&pipeline, ui_frame.clone())
        .map_err(|error| frame_error(window, format!("UI pipe creation failed: {error}")))?;
    ui_pipe.set_clear(CLEAR);
    compositor
        .add_pipe(ui_pipe)
        .map_err(|error| frame_error(window, format!("UI pipe registration failed: {error}")))?;
    *started = true;
    log_info!("presenting {}x{} dma-buf frames", image.width, image.height);

    let mut changes = Changes::default();
    let mut dirty = true;
    loop {
        if !window.is_open() {
            return Ok(SessionEnd::Closed);
        }
        if let Some(size) = changes.resize {
            log_info!("window resized to {}", size);
            return Ok(SessionEnd::Resized);
        }
        if !dirty && !changes.redraw {
            window.poll(None, &mut |event| changes.record(event))?;
            continue;
        }
        changes.redraw = false;

        // A single slot means the compositor must have finished reading
        // the previous frame before this one is recorded into it.
        while !window.buffer_free(SLOT) && window.is_open() && changes.resize.is_none() {
            window.poll(Some(RELEASE_POLL), &mut |event| changes.record(event))?;
        }
        if !window.is_open() {
            return Ok(SessionEnd::Closed);
        }
        if changes.resize.is_some() {
            return Ok(SessionEnd::Resized);
        }

        ui_frame.with(|list| paint(list, size));
        let sync_file = block_on(compositor.composite_frame(None))
            .map_err(|error| frame_error(window, format!("compositing failed: {error}")))?;
        let waited = wait_for_gpu(window, &sync_file, &mut changes);
        drop(sync_file);
        waited?;
        if !window.is_open() {
            return Ok(SessionEnd::Closed);
        }
        if changes.resize.is_some() {
            return Ok(SessionEnd::Resized);
        }

        // The callback is requested before the commit it belongs to, so
        // it fires once these pixels are on screen.
        window.request_frame()?;
        window.present(SLOT, &image)?;
        dirty = false;
    }
}

/// Records the launcher shell for a `size` surface.
fn paint(list: &mut DrawList, size: WndSize) {
    list.clear();
    let width = size.width as f32;
    let height = size.height as f32;
    if width <= 0.0 || height <= 0.0 {
        return;
    }
    list.rect(
        Rect::from_min_size(Offset::ZERO, width, height),
        CornerRadii::ZERO,
        Paint::fill(BACKDROP),
    );

    let field_width = (width * FIELD_WIDTH_RATIO)
        .clamp(MIN_FIELD_WIDTH, MAX_FIELD_WIDTH)
        .min((width - 2.0 * EDGE).max(0.0));
    let field_height = FIELD_HEIGHT.min((height - 2.0 * EDGE).max(0.0));
    if field_width <= 0.0 || field_height <= 0.0 {
        return;
    }
    let radius = FIELD_RADIUS.min(field_height * 0.5);
    let origin = Offset::new((width - field_width) * 0.5, (height - field_height) * 0.5);
    let field = Rect::from_min_size(origin, field_width, field_height);
    list.rect(field, CornerRadii::uniform(radius), Paint::fill(FIELD));

    let inset = FIELD_INSET
        .min(field_width * 0.25)
        .min(field_height * 0.25);
    if inset <= 0.0 {
        return;
    }
    let inner = Rect::from_min_size(
        Offset::new(origin.x + inset, origin.y + inset),
        field_width - 2.0 * inset,
        field_height - 2.0 * inset,
    );
    list.rect(
        inner,
        CornerRadii::uniform((radius - inset).max(0.0)),
        Paint::fill(FIELD_INNER),
    );
}

/// Builds a pipeline for a `size` target from the compositor's modifiers.
///
/// The linear layout is preferred when it is offered because it is the
/// most portable; if the driver cannot create the image with that single
/// modifier, every advertised one is tried instead.
fn create_pipeline(id: WndBackendId, size: WndSize, modifiers: &[u64]) -> Result<PipelineContext, WndError> {
    if size.width <= 0 || size.height <= 0 {
        return Err(WndError::backend(
            id,
            format!("window size {} cannot be rendered", size),
        ));
    }
    let linear = [0_u64];
    let requested: &[u64] = if modifiers.contains(&0) {
        &linear
    } else {
        modifiers
    };
    let width = size.width as u32;
    let height = size.height as u32;
    match PipelineContext::new(width, height, requested) {
        Ok(pipeline) => Ok(pipeline),
        Err(_) if requested.len() == 1 => PipelineContext::new(width, height, modifiers)
            .map_err(|error| WndError::backend(id, format!("the GPU target could not be built: {error}"))),
        Err(error) => Err(WndError::backend(
            id,
            format!("the GPU target could not be built: {error}"),
        )),
    }
}

/// Describes an exported render target as the image the window presents.
fn present_image(target: &RenderTarget) -> WndImage {
    WndImage {
        fd: target.dmabuf_fd.as_raw(),
        drm_format: target.drm_format,
        modifier: target.modifier,
        stride: target.stride,
        offset: target.offset,
        width: target.width as i32,
        height: target.height as i32,
    }
}

/// Creates a backend error attributed to `window`'s backend.
fn frame_error(window: &dyn Window, message: impl Into<alloc::string::String>) -> WndError {
    WndError::backend(window.backend(), message)
}

/// Blocks until the GPU finished the frame `sync_file` represents,
/// polling window events meanwhile.
fn wait_for_gpu(window: &mut dyn Window, sync_file: &OwnedFd, changes: &mut Changes) -> Result<(), WndError> {
    let id = window.backend();
    let mut descriptor = libc::pollfd {
        fd: sync_file.as_raw(),
        events: libc::POLLIN,
        revents: 0,
    };
    let mut waited = 0;
    loop {
        // SAFETY: `descriptor` points at a live `pollfd` holding an open,
        // pollable `sync_file` descriptor owned by the caller below, and
        // `nfds` is 1, matching the single entry of the array.
        let ready = unsafe { libc::poll(&mut descriptor, 1, GPU_POLL_SLICE_MS) };
        if ready > 0 {
            return Ok(());
        }
        waited += GPU_POLL_SLICE_MS;
        window.poll(Some(TimeDuration::ZERO), &mut |event| changes.record(event))?;
        if !window.is_open() || changes.resize.is_some() {
            return Ok(());
        }
        if waited >= GPU_SYNC_TIMEOUT_MS {
            return Err(WndError::backend(
                id,
                format!("the GPU did not finish the frame within {GPU_SYNC_TIMEOUT_MS} ms"),
            ));
        }
    }
}

/// Window events recorded while the loop renders or waits.
#[derive(Debug, Default)]
struct Changes {
    /// The size the compositor requested, if it changed.
    resize: Option<WndSize>,
    /// Whether the compositor asked for another frame.
    redraw: bool,
}

impl Changes {
    /// Records one window event.
    fn record(&mut self, event: WndEvent) {
        match event {
            WndEvent::Resized(size) => self.resize = Some(size),
            WndEvent::RedrawRequested => self.redraw = true,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use codevar_kuitui::paint::DrawVisitor;

    /// Collects the rectangles a display list paints.
    #[derive(Default)]
    struct RectSpy {
        rects: Vec<(Rect, CornerRadii)>,
    }

    impl DrawVisitor for RectSpy {
        fn rect(&mut self, rect: Rect, radii: CornerRadii, _paint: Paint) {
            self.rects.push((rect, radii));
        }
    }

    fn recorded(size: WndSize) -> Vec<(Rect, CornerRadii)> {
        let mut list = DrawList::new();
        paint(&mut list, size);
        let mut spy = RectSpy::default();
        for op in list.ops() {
            op.visit(&mut spy);
        }
        spy.rects
    }

    #[test]
    fn the_shell_covers_the_whole_surface() {
        let rects = recorded(WndSize::new(800, 600));
        assert_eq!(rects.len(), 3);
        let (backdrop, radii) = rects[0];
        assert_eq!((backdrop.left(), backdrop.top()), (0.0, 0.0));
        assert_eq!((backdrop.right(), backdrop.bottom()), (800.0, 600.0));
        assert!(radii.is_zero());
    }

    #[test]
    fn the_search_field_is_centered_and_rounded() {
        let rects = recorded(WndSize::new(800, 600));
        let (field, radii) = rects[1];
        assert_eq!(field.right() - field.left(), 800.0 * FIELD_WIDTH_RATIO);
        assert_eq!(field.bottom() - field.top(), FIELD_HEIGHT);
        assert_eq!(field.left(), (800.0 - 800.0 * FIELD_WIDTH_RATIO) * 0.5);
        assert_eq!(field.top(), (600.0 - FIELD_HEIGHT) * 0.5);
        assert_eq!(radii, CornerRadii::uniform(FIELD_RADIUS));
        let (inner, inner_radii) = rects[2];
        assert_eq!(inner.left(), field.left() + FIELD_INSET);
        assert_eq!(inner.top(), field.top() + FIELD_INSET);
        assert_eq!(inner_radii, CornerRadii::uniform(FIELD_RADIUS - FIELD_INSET));
    }

    #[test]
    fn a_tiny_window_still_paints_a_backdrop() {
        let rects = recorded(WndSize::new(24, 24));
        assert_eq!(rects.len(), 1);
        assert_eq!(rects[0].0.right(), 24.0);
    }
}
