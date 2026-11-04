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

//! The pipe model: the type-erased [`Pipe`], the owning [`PipeHandoff`]
//! handed to the compositor and the ordered [`PipeQueue`] whose order is
//! the mix order.
//!
//! The shape mirrors `codevar-gpu-vk`, with one difference the API forces:
//! a Vulkan image and view are copyable handles, so the queue used to
//! collect framebuffers into a `Vec`. A `wgpu-hal` framebuffer borrows the
//! texture and view it names, so it cannot outlive the renderer that owns
//! them; [`PipeQueue::for_each_framebuffer`] therefore hands each
//! framebuffer to a sink while the renderer is still borrowed.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;

use crate::offscreen::{OffscreenFramebuffer, PipeOffscreen, sanitize_weight};
use crate::pipe_ctx::{PipeCtx, PipeEntry, PipeEnv, PipeFuture};

/// One registered pipe: an async body plus the renderer it drives.
///
/// The layout is part of the module's contract — the entry pointer sits
/// at byte offset 0 and the erased renderer pointer at offset 8 on
/// 64-bit targets — and is asserted by the `const` checks below.
#[repr(C)]
pub struct Pipe {
    /// Type-erased entry point called once per frame.
    entry: PipeEntry,
    /// `*mut T` for the `PipeSource`/`PipeOffscreen` type `T` this
    /// pipe was created from.
    renderer: *mut (),
}

const _: () = assert!(core::mem::offset_of!(Pipe, entry) == 0);
const _: () = assert!(core::mem::offset_of!(Pipe, renderer) == core::mem::size_of::<PipeEntry>());
const _: () = assert!(core::mem::size_of::<Pipe>() == 2 * core::mem::size_of::<PipeEntry>());

impl Pipe {
    /// Builds a pipe for `renderer`, erasing its type behind
    /// [`PipeEntry`].
    ///
    /// The renderer must outlive the pipe; [`PipeHandoff`] upholds this
    /// by taking ownership of the renderer, which is why this
    /// constructor is not public.
    fn new<T: PipeSource>(renderer: &mut T) -> Self {
        Self {
            entry: trampoline::<T>,
            renderer: core::ptr::from_mut(renderer).cast::<()>(),
        }
    }

    /// The type-erased entry point of this pipe.
    #[inline]
    #[must_use]
    pub const fn entry(&self) -> PipeEntry {
        self.entry
    }

    /// The erased renderer pointer (only valid for the pipe's lifetime).
    #[inline]
    #[must_use]
    pub const fn renderer(&self) -> *mut () {
        self.renderer
    }
}

impl fmt::Debug for Pipe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipe")
            .field("entry", &self.entry)
            .field("renderer", &self.renderer)
            .finish()
    }
}

/// Per-frame async body of a registered pipe.
///
/// The compositor calls this once per frame with the pipe's renderer and
/// the frame's [`PipeCtx`]; the returned future records the pipe's
/// commands and may yield (its waker decides when it is polled again)
/// before returning a [`PipeOutcome`](crate::pipe_ctx::PipeOutcome).
pub trait PipeSource {
    /// Builds this pipe's future for one frame.
    fn pipe_entry<'a>(renderer: &'a mut Self, ctx: &'a mut PipeCtx) -> PipeFuture<'a>
    where
        Self: Sized;
}

/// Erases a pipe's type and its future's lifetime for the compositor.
///
/// # Safety
///
/// `env` must come from [`Pipe::new`] for the concrete `T`, and both
/// pointees must outlive the returned future: the compositor stores the
/// renderer in a [`PipeHandoff`] and the context in the pipe's active
/// slot, and it drops the future before either of them (the renderer is
/// only released at the next frame boundary).
fn trampoline<T: PipeSource>(env: PipeEnv) -> PipeFuture<'static> {
    // SAFETY: see the function's safety contract above; the two pointers
    // are not aliased elsewhere while the future is alive because the
    // compositor keeps exactly one active future per pipe.
    let renderer = unsafe { &mut *env.renderer.cast::<T>() };
    // SAFETY: same contract; the context is exclusively owned by the
    // pipe's active slot for the whole frame.
    let ctx = unsafe { &mut *env.ctx };
    let future: PipeFuture<'_> = T::pipe_entry(renderer, ctx);
    // SAFETY: the future borrows only `renderer` and `ctx`, which the
    // compositor keeps alive until the future is dropped; erasing the
    // lifetime here moves that obligation from the compiler to the drop
    // order documented in the safety contract above.
    unsafe { core::mem::transmute(future) }
}

/// Hands one pipe's framebuffers to a sink while the renderer is borrowed.
///
/// The sink receives the framebuffer and its sanitized mix weight, once
/// per valid framebuffer in index order. A borrowed framebuffer cannot be
/// stored past the borrow, so the compositor consumes it immediately.
type CollectFn = fn(*const (), &mut dyn for<'a> FnMut(OffscreenFramebuffer<'a>, f32));

/// Releases the renderer owned by a [`PipeHandoff`].
type DropFn = fn(*mut ());

/// A pipe queued with the compositor together with everything the frame
/// loop needs from it.
///
/// The handoff owns its renderer: dropping it releases the renderer
/// through the type-erased drop function, so a pipe registered with a
/// compositor keeps working after the caller's value would have gone out
/// of scope.
#[derive(Debug)]
pub struct PipeHandoff {
    /// Entry point and erased renderer of the pipe.
    pipe: Pipe,
    /// Per-frame framebuffer/weight visitor of the renderer.
    collect: CollectFn,
    /// Releases the renderer stored in `pipe`.
    drop_fn: DropFn,
}

impl PipeHandoff {
    /// Builds a handoff that takes ownership of `renderer`.
    #[must_use]
    pub fn from_renderer<T: PipeSource + PipeOffscreen>(renderer: T) -> Self {
        // The pipe records the address of the renderer, so the value
        // must live on the heap for as long as the handoff owns it;
        // `drop_trampoline` releases exactly this allocation.
        let mut boxed = Box::new(renderer);
        let pipe = Pipe::new(&mut *boxed);
        // Taking the raw pointer transfers ownership out of the `Box`,
        // so the local scope does not free what the pipe now points to.
        let renderer = Box::into_raw(boxed);
        debug_assert_eq!(renderer.cast::<()>(), pipe.renderer());
        Self {
            pipe,
            collect: collect_trampoline::<T>,
            drop_fn: drop_trampoline::<T>,
        }
    }

    /// The pipe registered with the compositor.
    #[inline]
    #[must_use]
    pub const fn pipe(&self) -> &Pipe {
        &self.pipe
    }

    /// Visits this pipe's framebuffers and weights for one frame.
    fn collect<S>(&self, sink: &mut S)
    where
        S: for<'a> FnMut(OffscreenFramebuffer<'a>, f32),
    {
        (self.collect)(self.pipe.renderer, sink);
    }
}

impl Drop for PipeHandoff {
    fn drop(&mut self) {
        (self.drop_fn)(self.pipe.renderer);
    }
}

/// Body of [`PipeHandoff::collect`] for the concrete renderer type.
fn collect_trampoline<T: PipeOffscreen>(
    renderer: *const (),
    sink: &mut dyn for<'a> FnMut(OffscreenFramebuffer<'a>, f32),
) {
    // SAFETY: the pointer is the renderer `PipeHandoff::from_renderer`
    // boxed for `T`; the handoff owns it and outlives the synchronous
    // collection below, which takes a shared borrow only.
    let renderer = unsafe { &*renderer.cast::<T>() };
    for index in 0..renderer.offscreen_framebuffer_count() {
        let Some(framebuffer) = renderer.offscreen_framebuffer(index) else {
            continue;
        };
        if !framebuffer.is_valid() {
            continue;
        }
        sink(framebuffer, sanitize_weight(renderer.mix_weight(index)));
    }
}

/// Body of [`PipeHandoff`]'s `Drop` for the concrete renderer type.
fn drop_trampoline<T>(renderer: *mut ()) {
    // SAFETY: the pointer is the `Box<T>` allocation that
    // `PipeHandoff::from_renderer` moved into the handoff, and this
    // function runs exactly once — from `PipeHandoff::drop`.
    drop(unsafe { Box::from_raw(renderer.cast::<T>()) });
}

/// One queued pipe plus its pending-removal flag.
struct QueueEntry {
    /// Pipe and owned renderer.
    handoff: PipeHandoff,
    /// Set by [`PipeQueue::mark_removed`]; the entry is dropped by
    /// [`PipeQueue::drain_removed`] once the GPU finished the frame that
    /// still referenced it.
    remove: bool,
}

/// Ordered registry of pipes: queue order is mix order.
///
/// The queue only changes between frames (registration and removal are
/// rejected while a frame is in progress), which is what lets the frame
/// loop index it positionally.
#[derive(Default)]
pub struct PipeQueue {
    /// Queued pipes in mix order.
    entries: Vec<QueueEntry>,
}

impl PipeQueue {
    /// Creates an empty queue.
    #[must_use]
    pub const fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Appends `handoff` and returns its index.
    pub fn push(&mut self, handoff: PipeHandoff) -> usize {
        self.entries.push(QueueEntry {
            handoff,
            remove: false,
        });
        self.entries.len() - 1
    }

    /// Number of queued pipes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no pipe is queued.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The handoff queued at `index`.
    #[must_use]
    pub fn handoff(&self, index: usize) -> Option<&PipeHandoff> {
        self.entries
            .get(index)
            .map(|entry| &entry.handoff)
    }

    /// The pipe queued at `index`.
    #[must_use]
    pub fn pipe(&self, index: usize) -> Option<&Pipe> {
        self.handoff(index).map(PipeHandoff::pipe)
    }

    /// Marks the pipe at `index` for removal at the next frame boundary.
    ///
    /// Returns `false` when no such pipe exists or it was already
    /// marked.
    pub fn mark_removed(&mut self, index: usize) -> bool {
        let Some(entry) = self.entries.get_mut(index) else {
            return false;
        };
        let fresh = !entry.remove;
        entry.remove = true;
        fresh
    }

    /// Whether any pipe is waiting to be removed.
    #[must_use]
    pub fn has_removed(&self) -> bool {
        self.entries.iter().any(|entry| entry.remove)
    }

    /// Drops every pipe marked for removal and returns them.
    ///
    /// Only call this after the frame that still referenced them retired
    /// (its fence was waited on), so releasing the renderers is safe.
    pub fn drain_removed(&mut self) -> Vec<PipeHandoff> {
        let mut retired = Vec::new();
        let mut index = 0;
        while index < self.entries.len() {
            if self.entries[index].remove {
                let entry = self.entries.remove(index);
                retired.push(entry.handoff);
            } else {
                index += 1;
            }
        }
        retired
    }

    /// Visits every pipe's framebuffers in queue order, tagged with the
    /// queue index of the pipe that contributed each one.
    ///
    /// The sink runs while each renderer is borrowed; see the module
    /// documentation for why the framebuffers are not collected.
    pub fn for_each_framebuffer<S>(&self, sink: &mut S)
    where
        S: for<'a> FnMut(usize, OffscreenFramebuffer<'a>, f32),
    {
        for (index, entry) in self.entries.iter().enumerate() {
            entry
                .handoff
                .collect(&mut |framebuffer, weight| sink(index, framebuffer, weight));
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod test_util {
    //! Renderers shared by the pipe and communication tests.

    use super::*;
    use crate::pipe_ctx::PipeOutcome;
    use core::marker::PhantomData;

    /// Renderer contributing at most one framebuffer with a given weight.
    pub(crate) struct DummyRenderer {
        /// The frame it hands to the mix pass, if any.
        pub(crate) framebuffer: Option<OffscreenFramebuffer<'static>>,
        /// Weight reported for that framebuffer.
        pub(crate) weight: f32,
    }

    impl DummyRenderer {
        /// A pipe whose fabricated (but structurally valid) 64×64 target
        /// passes the collector's checks without a live device.
        pub(crate) fn with_framebuffer(weight: f32) -> Self {
            Self {
                framebuffer: Some(OffscreenFramebuffer {
                    format: wgpu_types::TextureFormat::Bgra8Unorm,
                    width: 64,
                    height: 64,
                    marker: PhantomData,
                }),
                weight,
            }
        }

        /// A pipe that contributes no framebuffers.
        pub(crate) fn empty() -> Self {
            Self {
                framebuffer: None,
                weight: 1.0,
            }
        }
    }

    impl PipeSource for DummyRenderer {
        fn pipe_entry<'a>(_renderer: &'a mut Self, _ctx: &'a mut PipeCtx) -> PipeFuture<'a> {
            Box::pin(core::future::ready(Ok(PipeOutcome::Keep)))
        }
    }

    impl PipeOffscreen for DummyRenderer {
        fn offscreen_framebuffer_count(&self) -> usize {
            usize::from(self.framebuffer.is_some())
        }

        fn offscreen_framebuffer(&self, index: usize) -> Option<OffscreenFramebuffer<'_>> {
            if index == 0 { self.framebuffer } else { None }
        }

        fn mix_weight(&self, _index: usize) -> f32 {
            self.weight
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    //! Pipe dispatch, queue order, retirement and framebuffer visitation.

    use super::test_util::DummyRenderer;
    use super::*;

    #[test]
    fn pipe_dispatches_to_the_registered_type() {
        let handoff = PipeHandoff::from_renderer(DummyRenderer::empty());
        let expected: PipeEntry = trampoline::<DummyRenderer>;
        assert!(core::ptr::fn_addr_eq(handoff.pipe().entry(), expected));
        assert_eq!(core::mem::offset_of!(Pipe, entry), 0);
        assert_eq!(
            core::mem::offset_of!(Pipe, renderer),
            core::mem::size_of::<PipeEntry>()
        );
    }

    #[test]
    fn pipe_queue_registers_and_retires_in_order() {
        let mut queue = PipeQueue::new();
        let first = queue.push(PipeHandoff::from_renderer(DummyRenderer::empty()));
        let second = queue.push(PipeHandoff::from_renderer(DummyRenderer::empty()));
        assert_eq!((first, second), (0, 1));
        assert_eq!(queue.len(), 2);

        assert!(queue.mark_removed(0));
        assert!(!queue.mark_removed(0), "already marked");
        assert!(!queue.mark_removed(7), "no such pipe");
        assert!(queue.has_removed());

        let retired = queue.drain_removed();
        assert_eq!(retired.len(), 1);
        assert_eq!(queue.len(), 1);
        assert!(!queue.has_removed());
        assert!(queue.pipe(0).is_some());
        assert!(queue.pipe(1).is_none());
    }

    #[test]
    fn pipe_queue_accepts_unlimited_pipes() {
        let mut queue = PipeQueue::new();
        // Test that we can add many pipes without hitting a capacity limit
        for _ in 0..100 {
            queue.push(PipeHandoff::from_renderer(DummyRenderer::empty()));
        }
        assert_eq!(queue.len(), 100);
    }

    #[test]
    fn for_each_framebuffer_visits_in_order_and_sanitizes_weights() {
        let mut queue = PipeQueue::new();
        queue.push(PipeHandoff::from_renderer(DummyRenderer::with_framebuffer(0.25)));
        queue.push(PipeHandoff::from_renderer(DummyRenderer::with_framebuffer(
            f32::NAN,
        )));
        queue.push(PipeHandoff::from_renderer(DummyRenderer::with_framebuffer(2.0)));
        queue.push(PipeHandoff::from_renderer(DummyRenderer::empty()));

        let mut visits = Vec::new();
        queue.for_each_framebuffer(&mut |index, framebuffer, weight| {
            assert!(framebuffer.is_valid());
            assert_eq!((framebuffer.width, framebuffer.height), (64, 64));
            visits.push((index, weight));
        });

        assert_eq!(visits, [(0, 0.25), (1, 0.0), (2, 1.0)]);
    }
}
