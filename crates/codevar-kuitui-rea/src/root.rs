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

//! The root.
//!
//! [`Root`] owns everything a tree needs to come to life: the host, the fiber
//! arena, the committed tree, the mailbox of dispatches and the lanes that
//! still have work. It is the counterpart of React's `FiberRoot` and work
//! loop, with one deliberate difference: it has no scheduler of its own. The
//! host decides when a frame is worth flushing, and [`Root::flush_slice`] gives
//! control back between units of work so a render can stay inside the host's
//! budget.
//!
//! # The flush loop
//!
//! ```text
//! loop {
//!     drain the mailbox into pending lanes
//!     next = the lanes with the highest priority
//!     if there are none: return
//!     render on `next`, commit, repeat
//! }
//! ```
//!
//! A dispatch never renders on its own. It queues the update and signals the
//! mailbox; draining the mailbox is what turns a wake into pending lanes and
//! marks them on the fiber that owns the queue and on its ancestors. Draining
//! between passes is what lets an update dispatched by a layout effect — or by
//! one component while another renders — take part in the same flush as the
//! work that caused it. An update a component dispatches to *itself* while it
//! renders never reaches the mailbox at all; see below.
//!
//! # Time slicing
//!
//! A render whose lanes are all speculative — transitions, retries and the
//! other work no input is waiting on — may be stopped between units of work. A
//! unit is one fiber begun plus the chain of ancestors that completes with it,
//! so a component is never left half-rendered. [`Root::flush_slice`] asks its
//! caller before each unit; when the caller says the slice is over, the render
//! stays in flight, the work-in-progress tree reaches no host, and the next
//! slice resumes from the fiber that had not started. A render that covers a
//! blocking lane — sync, continuous input, default or gesture work — is never
//! stopped, even when its caller asks; [`Root::flush`] never stops at all.
//!
//! Work that arrives while a render is in flight only replaces it when it is
//! strictly more urgent than the lanes the render covers; default work does
//! not interrupt a transition. When work does replace the render, the tree in
//! flight is thrown away: its fibers are freed and its update queues return to
//! the state the last commit left, so the next render replays them from the
//! committed tree. A lane a render could not cover, and a lane that arrived
//! while it rendered, stay pending for a later pass.
//!
//! # Starvation
//!
//! High-priority work can keep a low-priority lane pending forever: a stream of
//! input interrupts every transition render before it commits. The root
//! therefore records when the clock first sees a lane pending and marks the
//! lane *expired* once [`Lanes::expiration_delay`] has passed. A render that
//! covers an expired lane is never time sliced, so the starved work finishes in
//! one slice instead of being interrupted again. Retry lanes do not expire —
//! React gates that behind `enableRetryLaneExpiration`, which is off — because a
//! retry has to be time sliced to unwrap an uncached promise.
//!
//! # Render phase updates
//!
//! A component that writes to its own state while it renders is re-rendered in
//! place, before its element reaches the reconciler: the pass reruns the
//! component until no pass queues another update, so the intermediate element
//! is never mounted and the commit applies the tree the final value produced.
//! React calls these *render phase updates* and bounds the loop at 25 extra
//! passes; the work pass does the same. Where React throws "Too many
//! re-renders" and unmounts the tree, Rea stops the loop and drops the updates
//! of the last pass — they were queued for a pass the limit did not allow —
//! and commits the tree the last allowed pass described.
//!
//! A dispatch to *another* component's state during a render is an ordinary
//! update: it signals the mailbox and the flush loop renders that component in
//! its own pass.
//!
//! # Lanes and what a commit leaves behind
//!
//! A render is started for one lane set and covers exactly that set. An update
//! in the queue whose lane the render does not cover stays in the base queue
//! (see [`crate::hooks`]) and puts its lane back on the fiber; bubbling the
//! fiber lanes to the root is what turns "this render could not apply
//! everything" into `finishedWork.lanes | finishedWork.child_lanes`. The root
//! makes that set the pending lanes *before* the commit mutates the host, the
//! way `markRootFinished` does, so a lane is never both rendered and pending.
//!
//! Idle, offscreen and deferred work is picked only when no other lane is
//! pending. The root makes the lanes the render could not cover pending
//! *before* the commit mutates the host, the way `markRootFinished` does, so a
//! lane is never both rendered and pending.
//!
//! # Passive effects
//!
//! The passive effects of a commit do not run in its layout pass. They run in
//! one of three places:
//!
//! * [`Root::flush_passive`], which the host calls once the frame that
//!   produced them is presented;
//! * the start of the next commit, which must run them before it mutates the
//!   host, so their destroy pass still visits the tree that created them
//!   (`completeRoot` in React); or
//! * the end of the same commit, when the render covered the sync-update
//!   group ([`Lanes::SYNC_UPDATE`]), so a discrete update is fully observable
//!   when the caller regains control.
//!
//! # Work the render skips
//!
//! A fiber whose pending description is the one it already committed and whose
//! lanes the render does not cover is not rendered again: it keeps its
//! children, and the pass clones them into the finished tree so both trees own
//! a copy of the subtree. The bailout stops outright when `child_lanes` holds
//! no lane of the render, which skips a whole subtree with nothing pending in
//! it. A parent that builds a new element for a component defeats this, the
//! way React re-renders a child whose props object changed.
//!
//! # Suspension
//!
//! A component suspends by recording a wakeable instead of rendering (see
//! [`RenderCtx::suspend`]). A suspension that reaches the shell — nothing above
//! it can show a substitute — throws the render away and parks the lanes it
//! covered: nothing is committed, and the host keeps showing the last tree that
//! completed. A parked lane is still pending but no flush can make progress on
//! it, so [`Root::has_pending_work`] ignores it and [`Root::next_lanes`] skips
//! it, which is React's `getNextLanes` with a non-empty `suspendedLanes`.
//!
//! The root subscribes the wakeable before it parks the lanes. When the work it
//! names resolves, it *pings*: the lanes go into the root's pinged set, which
//! is the only way a still-suspended lane becomes a candidate for a render, and
//! the mailbox entry the ping raises is what tells the host that the tree has
//! something to try. A render that suspends in the same place clears the lanes
//! from the pinged set as it parks them again, so a wakeable that cannot help
//! does not cause a retry loop. An update on any lane that is not idle clears
//! the parked set outright — `markRootUpdated` assumes the update may carry the
//! data the tree was waiting for — and the next flush tries the parked work
//! again. Parked lanes lose their expiration clock: the wait is for data, not
//! for the CPU, so starvation protection has nothing to measure.
//!
//! A suspension below a [`Element::suspense`] boundary does not reach
//! the root: the boundary catches it, the pass throws away the primary tree it
//! was building and renders the fallback in its place, and the commit shows the
//! fallback. The primary tree stays mounted but hidden, so the host does not
//! have to build it again when the boundary returns to it, and the lanes the
//! interrupted render left below the boundary stay pending — that is what lets
//! a later update retry the tree.
//!
//! The boundary does not wait for that update. The commit subscribes a retry
//! listener to the wakeable the boundary caught, and when the work resolves the
//! listener leaves the boundary in the mailbox; the next drain claims one of
//! the four retry lanes, rotating them so two retries never share a lane, and
//! marks the boundary with it. The render that lane covers re-renders the
//! boundary, which reconciles its primary tree again and deletes the fallback
//! if the work it needed is there. A retry that suspends in the same place
//! subscribes the new wakeable and stays on its fallback, so nothing retries in
//! a loop.
//!
//! A fallback is not always worth committing. A boundary that has to replace
//! content the user is looking at would make that content disappear, so a
//! render whose lanes are transitions only — the lanes that are allowed to wait
//! — parks them and commits nothing, leaving the previous screen in place until
//! the work resolves. A boundary that mounts hides nothing, and neither does one
//! inside a tree that is already out of sight, so their fallbacks commit at
//! once; that is React's `RootSuspendedWithDelay` against `RootSuspended`.
//!
//! A fallback that would take visible content away from any other render is not
//! committed at once either. Its finished tree is held until the next Just
//! Noticeable Difference boundary after the newest update the render covers —
//! React's `jnd` — so a user who already waited does not get a loading state
//! the next instant. A wait of a few milliseconds is not worth holding for, and
//! a root with no clock cannot measure one, so both commit the placeholder at
//! once.
//!
//! A retry is throttled as well. A tree that retries through several levels —
//! the data one boundary waited for arrives and reveals another boundary that
//! still waits — would otherwise replace one loading state with the next within
//! a frame. A render whose lanes are retries only and that has to show a
//! fallback therefore waits until [`FALLBACK_THROTTLE_MS`] has passed since the
//! last fallback that hid content: the finished tree is held, its lanes are
//! parked, and [`Root::flush_slice`] commits the tree once its clock reaches
//! the deadline. Work that arrives before then replaces the held tree, and
//! [`Root::flush`], which has no clock, commits it at once.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::slice;

use crate::commit::Commit;
use crate::element::Element;
use crate::fiber::{Arena, Fiber, FiberId, FiberTag, Flags};
use crate::hooks::{MailboxHandle, RetryRequest, mailbox};
use crate::host::Host;
use crate::lane::{LANE_COUNT, Lanes};
use crate::reconcile::reconcile_children;
use crate::wakeable::Wakeable;
use crate::work::{ExitStatus, Progress, WorkLoop};

/// A render in flight.
///
/// The work-in-progress tree is complete when `next` is `None`; a render that
/// yielded leaves the fiber it was about to begin, so the next slice resumes
/// exactly there.
struct Render {
    /// The work-in-progress root.
    root: FiberId,
    /// The next fiber to begin, or `None` when the tree is complete.
    next: Option<FiberId>,
    /// The lanes this render covers.
    lanes: Lanes,
    /// What the pass found out about the tree.
    ///
    /// A render can stop between units of work, so the outcome is kept here
    /// rather than in the pass: React keeps `workInProgressRootExitStatus` on
    /// the root for the same reason, and a slice that resumed with a fresh
    /// status would commit a fallback the whole render wanted to avoid.
    exit: ExitStatus,
}

/// A finished tree the root holds back until its commit is due.
///
/// This is React's `root.finishedWork` together with the `timeoutHandle`
/// `finishConcurrentRender` schedules for it: the render completed, but showing
/// the tree now would take content away too soon — replacing one loading state
/// with another, or a screen the user is reading with a placeholder — so the
/// commit waits until the host's clock reaches `deadline`. The fibers stay in
/// the arena meanwhile; a new render throws them away (React's
/// `prepareFreshStack` cancels the handle).
struct Held {
    /// The finished work-in-progress root.
    root: FiberId,
    /// The lanes the render covered.
    lanes: Lanes,
    /// When the commit is due, on the host's clock.
    deadline: u64,
}

/// A mounted tree.
///
/// The root owns the host and the fibers and turns dispatches into renders. It
/// is single-threaded: the host, the fibers and every state handle share
/// `Rc`s, so a `Root` stays on the thread that created it.
///
/// ```ignore
/// let mut root = Root::new(host, container);
/// root.render(element);   // schedules
/// root.flush();           // renders and commits what is pending
/// root.flush_passive();   // once the frame is presented
/// ```
pub struct Root<H: Host> {
    /// The host the nodes live in.
    host: H,
    /// The arena that owns the fibers.
    arena: Arena<H>,
    /// The committed tree.
    current: FiberId,
    /// The element the root renders.
    element: Option<Element<H>>,
    /// The mailbox every dispatch signals.
    mailbox: MailboxHandle,
    /// The counter of hook ids.
    ids: u64,
    /// The lanes that still have work.
    pending_lanes: Lanes,
    /// The lanes a render parked on a suspension.
    ///
    /// A lane in this set is still pending, but the tree cannot make progress
    /// on it: what it renders needs data that is not there, so a render would
    /// suspend in the same place. React keeps the same set on the root and
    /// takes it out of `getNextLanes`.
    suspended_lanes: Lanes,
    /// The parked lanes whose work has resolved since they were parked.
    ///
    /// A lane stays suspended when its work pings: the retry may suspend in the
    /// same place, and parking it again from a set it is still in would say
    /// nothing. Instead the pinged set is what lets [`Root::next_lanes`] pick a
    /// still-suspended lane, exactly as React's `getNextLanes` consults
    /// `pingedLanes` when no unsuspended work is left.
    pinged_lanes: Lanes,
    /// The wakeables the root subscribed to, with the lanes they were
    /// subscribed for.
    ///
    /// This is React's `pingCache`. A lane can suspend on the same work more
    /// than once — the render retries and the component reads the same missing
    /// value — so the root subscribes once per pair and forgets the wakeable
    /// when it pings, because work that completed is never thrown again.
    ping_cache: Rc<RefCell<Vec<(Wakeable, Lanes)>>>,
    /// When each pending lane expires, indexed by lane bit. `None` until the
    /// root first sees the lane pending, and again once the lane is rendered.
    expiration_times: [Option<u64>; LANE_COUNT],
    /// When each pending lane's newest update was requested, indexed by lane
    /// bit.
    ///
    /// This is React's `root.eventTimes`, which a delayed fallback measures its
    /// wait from: a user who has already waited should not get a loading state
    /// the next instant. `None` is React's `NoTimestamp` — no clock has dated
    /// the lane yet.
    event_times: [Option<u64>; LANE_COUNT],
    /// The lanes that have been pending past their expiration delay.
    expired_lanes: Lanes,
    /// The retry lane the next boundary retry claims.
    ///
    /// This is React's `nextRetryLane`. The four retry lanes rotate, so a retry
    /// that follows another does not share its lane and the two are rendered
    /// separately instead of entangled.
    next_retry_lane: Lanes,
    /// The render in flight, if a slice stopped before the tree completed.
    render: Option<Render>,
    /// The finished tree whose commit waits for its deadline.
    held: Option<Held>,
    /// When the last fallback replaced content the user could see, on the
    /// host's clock.
    ///
    /// This is React's `globalMostRecentFallbackTime`. It stays `None` until a
    /// commit that hid a primary tree runs with a clock: without a moment to
    /// measure from there is nothing a later fallback could be throttled
    /// against.
    fallback_time: Option<u64>,
    /// Work that arrived while a render was in flight. Its lane is pending but
    /// no fiber of that render necessarily carries it, so the completion has to
    /// account for it on its own.
    interleaved: Lanes,
    /// The finished tree whose passive effects have not run yet.
    passive: Option<FiberId>,
}

impl<H: Host> Root<H> {
    /// Creates a root that attaches its tree to `container`.
    ///
    /// `container` is a node the host already owns — the window's root widget,
    /// a vector index in a test — and becomes the parent of the top-level
    /// element the root renders.
    #[must_use]
    pub fn new(host: H, container: H::Node) -> Self {
        let mut arena = Arena::new();
        let mut fiber = Fiber::new(FiberTag::Root);
        fiber.state_node = Some(container);
        let current = arena.alloc(fiber);
        Self {
            host,
            arena,
            current,
            element: None,
            mailbox: mailbox(),
            ids: 0,
            pending_lanes: Lanes::NONE,
            suspended_lanes: Lanes::NONE,
            pinged_lanes: Lanes::NONE,
            ping_cache: Rc::new(RefCell::new(Vec::new())),
            expiration_times: [None; LANE_COUNT],
            event_times: [None; LANE_COUNT],
            expired_lanes: Lanes::NONE,
            next_retry_lane: Lanes::RETRY_FIRST,
            render: None,
            held: None,
            fallback_time: None,
            interleaved: Lanes::NONE,
            passive: None,
        }
    }

    /// The host the tree was built in.
    #[must_use]
    pub fn host(&self) -> &H {
        &self.host
    }

    /// The host, mutably.
    pub fn host_mut(&mut self) -> &mut H {
        &mut self.host
    }

    /// Whether a flush would find work to do.
    ///
    /// A dispatch or a ping that only reached the mailbox counts: it becomes
    /// pending work when [`Root::flush`] drains the mailbox. A finished tree
    /// the root holds back until its deadline counts too, because the host has
    /// to keep giving the root the clock until the commit is due. A lane the
    /// tree suspended on does not count while nothing has pinged it, and neither
    /// does idle work that waits behind such a lane: a render would suspend in
    /// the same place, or would show the parked work as if it had finished, so
    /// a host that keeps flushing while this is true cannot spin on work it
    /// cannot make progress on. React reaches the same conclusion in
    /// `getNextLanes`, which returns no lanes for exactly those sets; the
    /// dispatch or ping that unblocks the work is a mailbox entry the moment it
    /// arrives, so the host sees it.
    #[must_use]
    pub fn has_pending_work(&self) -> bool {
        self.held.is_some() || !self.next_lanes().is_empty() || !self.mailbox.borrow().is_empty()
    }

    /// Schedules `element` as the tree of the root.
    ///
    /// The element replaces the previous one on the next flush. A dispatch that
    /// arrives from a state handle in the meantime is merged with it, on the
    /// default lane unless the dispatch names a lane of its own.
    pub fn render(&mut self, element: Element<H>) {
        self.element = Some(element);
        self.mark_update(self.current, Lanes::DEFAULT);
    }

    /// Runs `scope` with every dispatch inside it marked as a transition.
    ///
    /// This is React's `startTransition`. Nothing renders when the call
    /// returns: the dispatches the scope made are queued on a transition lane
    /// and wait for the host to flush, so the committed tree stays on screen
    /// until the new one is ready. Transition work renders after the urgent
    /// lanes, and a transition that waited past its expiration delay finishes
    /// in one slice so urgent work cannot postpone it forever.
    ///
    /// The scopes of one event share a lane, so the updates they make render
    /// together. The root clears that cache when it starts working — the
    /// concurrent work loop is what says the event is over — so a transition
    /// started by a later event takes the next lane of the rotation and never
    /// waits behind an older one. A nested scope reuses the lane of the scope
    /// around it, rather than starting a transition of its own.
    pub fn start_transition<R>(&self, scope: impl FnOnce() -> R) -> R {
        crate::hooks::start_transition(&self.mailbox, scope)
    }

    /// Renders and commits every pending lane, most urgent first.
    ///
    /// The call returns when no lane is pending and no dispatch is left in the
    /// mailbox, or when every lane that is still pending is parked on a
    /// suspension. Work that a layout effect or a component dispatches during
    /// the call is picked up by the loop, so a caller that sets state from an
    /// effect sees the result before `flush` returns. Passive effects follow
    /// their own rule; see the module documentation.
    ///
    /// The call supplies no clock, so a finished tree the root would otherwise
    /// hold back for a fallback throttle commits at once: there is no moment to
    /// schedule the wait from. A host that wants the throttle uses
    /// [`Root::flush_slice`].
    pub fn flush(&mut self) {
        self.flush_loop(None, &mut || false);
    }

    /// Renders and commits the pending lanes, giving control back between
    /// units of work.
    ///
    /// `now` is the host's monotonic clock in milliseconds. The root uses it to
    /// mark the pending lanes that have waited past [`Lanes::expiration_delay`]
    /// as expired, and a render that covers an expired lane is never stopped
    /// for the caller's budget: starved work finishes in this slice. The clock
    /// also decides when a finished tree the root holds back for a fallback may
    /// commit — one throttled after another fallback, or one delayed until the
    /// next noticeable difference; [see the module documentation](self#suspension).
    ///
    /// `should_yield` is asked before every unit of a render whose lanes are
    /// neither blocking nor expired, and a `true` answer leaves the render in
    /// flight: nothing the slice built has reached the host, and the next call
    /// resumes the render at the fiber it had not started. A render that covers
    /// a blocking lane ignores the callback and always completes.
    ///
    /// The return value says whether work is left: `false` means nothing is
    /// pending, the tree committed everything it was asked to cover. A held
    /// tree counts as left, so a host keeps calling until the commit is due.
    /// Parked lanes do not count as left — the next call has nothing to do on
    /// them until an update unblocks them — so a parked tree also returns
    /// `false`.
    pub fn flush_slice(&mut self, now: u64, mut should_yield: impl FnMut() -> bool) -> bool {
        // The mailbox is drained first so the lanes this call sees are the ones
        // the host's clock starts the expiration of; React marks them from
        // `ensureRootIsScheduled`, after the update has marked the root.
        self.drain_mailbox(Some(now));
        self.mark_starved_lanes(now);
        self.flush_loop(Some(now), &mut should_yield)
    }

    /// Renders and commits the pending lanes until none is left or a slice
    /// stops the render.
    ///
    /// `now` is the host's clock, when the host has one. `None` is the clockless
    /// [`Root::flush`], which cannot wait: a held tree commits at once.
    fn flush_loop(&mut self, now: Option<u64>, should_yield: &mut dyn FnMut() -> bool) -> bool {
        // The event that scheduled this work is over, so a transition started
        // from now on belongs to the next one and takes the next lane of the
        // rotation. This is where React's `performConcurrentWorkOnRoot` clears
        // `currentEventTransitionLane`.
        self.mailbox.borrow_mut().end_event();
        loop {
            self.drain_mailbox(now);
            let lanes = self.next_lanes();
            if !lanes.is_empty() {
                if !self.perform(lanes, now, should_yield) {
                    return true;
                }
                continue;
            }
            // Nothing is runnable. A held tree commits once its deadline
            // passed; until then it is work this root still owes the host, the
            // way React's timeout handle is a callback it still owes.
            if !self.commit_held(now) {
                return self.held.is_some();
            }
        }
    }

    /// Commits the held tree once the host's clock reaches its deadline.
    ///
    /// Returns whether a tree was committed. This is the `commitRoot` callback
    /// React hands to `scheduleTimeout`: the tree is finished, so nothing is
    /// rendered again — the commit simply applies it. A host that supplies no
    /// clock cannot schedule the wait, so the tree commits at once.
    fn commit_held(&mut self, now: Option<u64>) -> bool {
        let Some(held) = self.held.take() else {
            return false;
        };
        if now.is_some_and(|now| now < held.deadline) {
            self.held = Some(held);
            return false;
        }
        self.arena.end_render();
        self.commit(held.root, held.lanes, now);
        true
    }

    /// Marks the lanes that have been pending past their expiration delay.
    ///
    /// This is React's `markStarvedLanesAsExpired`. The first time the host's
    /// clock sees a lane pending the root records when that lane expires; once
    /// the clock passes that time the lane is expired, and a render that covers
    /// it runs without time slicing, so urgent work cannot keep postponing it
    /// forever. Retry lanes are left out, exactly as React does with
    /// `enableRetryLaneExpiration` off: a retry has to be time sliced to unwrap
    /// an uncached promise.
    fn mark_starved_lanes(&mut self, now: u64) {
        // A lane marked without a clock — the element of `Root::render`, or an
        // update that a clockless flush drained — is dated the first time the
        // host's clock sees it pending. That is the closest rea gets to React's
        // request time, and a delayed fallback measures its wait from it.
        for lane in self.pending_lanes.iter() {
            if let Some(index) = lane.index() {
                let index = index as usize;
                if self.event_times[index].is_none() {
                    self.event_times[index] = Some(now);
                }
            }
        }
        let mut expired = self.expired_lanes;
        // A parked lane waits for data instead of the CPU, so it is not
        // starving and React does not arm its clock (`markStarvedLanesAsExpired`
        // skips the suspended lanes that were not pinged).
        let starving = self.pending_lanes & !Lanes::RETRY & !self.suspended_lanes;
        for lane in starving.iter() {
            let Some(index) = lane.index() else {
                continue;
            };
            let index = index as usize;
            match self.expiration_times[index] {
                // Assumes the clock is monotonic.
                None => {
                    self.expiration_times[index] = lane
                        .expiration_delay()
                        .map(|delay| now.saturating_add(delay));
                }
                Some(at) if at <= now => expired.insert(lane),
                Some(_) => {}
            }
        }
        self.expired_lanes = expired;
    }

    /// Runs the passive effects of the last commit, then flushes the work they
    /// scheduled.
    ///
    /// Returns whether there were passive effects to run. A host calls this
    /// once the frame that produced them is presented.
    pub fn flush_passive(&mut self) -> bool {
        let flushed = self.run_passive();
        self.flush();
        flushed
    }

    /// The lanes the next render must cover.
    ///
    /// This is React's `getNextLanes`. Unblocked non-idle work comes first, then
    /// non-idle work whose suspension pinged; only when no non-idle lane is
    /// pending at all does the root look at the idle, offscreen and deferred
    /// lanes, by the same two steps. A parked lane that has not pinged is not a
    /// candidate — a render would suspend in the same place — and idle work
    /// does not overtake non-idle work even while that work is parked, because
    /// a render that skipped the parked lane would commit the tree around a
    /// suspension that is still unresolved. When nothing qualifies the root
    /// returns no lanes. When a render is in flight, the work that arrived
    /// since it started replaces it only when strictly more urgent — and
    /// default work never interrupts a transition. That is React's rule in
    /// `getNextLanes`, which returns the lanes of the render in flight for
    /// everything else, so a render that has begun is not thrown away for equal
    /// or lower priority work.
    fn next_lanes(&self) -> Lanes {
        let pending = self.pending_lanes;
        if pending.is_empty() {
            return Lanes::NONE;
        }
        let non_idle = pending & Lanes::NON_IDLE;
        let next = if non_idle.is_empty() {
            let unblocked = pending & !self.suspended_lanes;
            if unblocked.is_empty() {
                self.pinged_lanes
            } else {
                unblocked
            }
        } else {
            let unblocked = non_idle & !self.suspended_lanes;
            if unblocked.is_empty() {
                non_idle & self.pinged_lanes
            } else {
                unblocked
            }
        }
        .render_lanes();
        if next.is_empty() {
            return Lanes::NONE;
        }
        let Some(render) = &self.render else {
            return next;
        };
        if next == render.lanes {
            return next;
        }
        let more_urgent = next
            .highest()
            .is_strictly_higher_than(render.lanes.highest());
        let default_over_transition =
            next.highest() == Lanes::DEFAULT && render.lanes.intersects(Lanes::TRANSITION);
        if more_urgent && !default_over_transition {
            next
        } else {
            render.lanes
        }
    }

    /// Renders `lanes` to completion, then commits the finished tree.
    ///
    /// Returns whether the render completed. A slice of non-blocking lanes
    /// stops between units when `should_yield` asks it to and leaves the render
    /// for the next call; a blocking render is never stopped. A render that
    /// suspends never completes: the tree is thrown away, nothing reaches the
    /// host, and the lanes are parked so no later flush repeats the attempt
    /// until something they wait for changed.
    fn perform(&mut self, lanes: Lanes, now: Option<u64>, should_yield: &mut dyn FnMut() -> bool) -> bool {
        if !self
            .render
            .as_ref()
            .is_some_and(|render| render.lanes == lanes)
        {
            self.prepare_fresh(lanes);
        }
        let Some(root) = self.render.as_ref().map(|render| render.root) else {
            return true;
        };
        let mut next = self
            .render
            .as_ref()
            .and_then(|render| render.next);
        // `shouldTimeSlice`: a render that covers blocking or expired work
        // runs to the end even when the host asks for the core.
        let sliceable = !lanes.intersects(Lanes::BLOCKING) && !lanes.intersects(self.expired_lanes);
        let mut suspended = None;
        let (exit, caught) = {
            let mut work_loop = WorkLoop::new(
                &mut self.host,
                &mut self.arena,
                Rc::clone(&self.mailbox),
                &mut self.ids,
                lanes,
            );
            while let Some(fiber) = next {
                if sliceable && should_yield() {
                    break;
                }
                match work_loop.step(fiber, root) {
                    Progress::Next(child) => next = Some(child),
                    Progress::Done => next = None,
                    // Nothing below the shell captured the suspension, so
                    // there is no fiber to render next: this is React's
                    // `RootSuspendedAtTheShell`, which unwinds to the root and
                    // leaves the finished tree uncommitted.
                    Progress::Suspended(wakeable) => {
                        suspended = Some(wakeable);
                        next = None;
                        break;
                    }
                }
            }
            (work_loop.exit_status(), work_loop.take_caught())
        };
        if let Some(render) = self.render.as_mut() {
            // A slice resumes the pass with a new work loop, so the outcome
            // accumulates on the render the way React keeps it on the root.
            render.exit = render.exit.max(exit);
        }
        if let Some(wakeable) = suspended {
            // React subscribes a suspension with no boundary as well: the lanes
            // are about to be parked, and the work resolving is the only thing
            // that can make them runnable again.
            self.attach_ping_listener(&wakeable, lanes);
            self.render = None;
            self.arena.abandon_render();
            self.interleaved = Lanes::NONE;
            self.mark_suspended(lanes);
            return true;
        }
        if next.is_some() {
            if let Some(render) = self.render.as_mut() {
                render.next = next;
            }
            return false;
        }
        let exit = self
            .render
            .as_ref()
            .map_or(ExitStatus::InProgress, |render| render.exit);
        if exit == ExitStatus::Suspended && includes_only_retries(lanes) {
            self.mark_suspended(lanes);
            if self.defer_retry(root, lanes, now) {
                return true;
            }
        }
        if exit == ExitStatus::SuspendedWithDelay {
            // React's `finishConcurrentRender`, case `RootSuspendedWithDelay`:
            // the lanes park either way; what follows decides whether anything
            // commits.
            self.mark_suspended(lanes);
            if includes_only_transitions(lanes) {
                // A transition that had to hide content keeps the previous
                // screen instead of committing a placeholder, and waits for the
                // data with no timeout at all. Nothing commits, so the retry
                // queue of the boundary is thrown away with the tree: the root
                // subscribes the work itself — React's `attachPingListener` in
                // `throwException` — and the ping is what makes the parked
                // lanes candidates again.
                self.render = None;
                self.arena.abandon_render();
                self.interleaved = Lanes::NONE;
                for wakeable in caught {
                    self.attach_ping_listener(&wakeable, lanes);
                }
                return true;
            }
            // A render that would hide content has to wait a little before it
            // replaces the screen, unless the wait is not worth taking.
            if self.defer_delay(root, lanes, now) {
                return true;
            }
        }
        self.render = None;
        self.arena.end_render();
        self.commit(root, lanes, now);
        true
    }

    /// Parks `lanes` until an update or a ping makes them runnable again.
    ///
    /// This is React's `markRootSuspended`: the lanes join the suspended set
    /// and lose their expiration clock, because the tree waits for data and not
    /// for the CPU. The lanes also lose their pinged mark: a retry that
    /// suspended in the same place has not been unblocked by the ping it was
    /// given, so the next render must not pick them again without new evidence.
    fn mark_suspended(&mut self, lanes: Lanes) {
        self.suspended_lanes.insert(lanes);
        self.pinged_lanes.remove(lanes);
        for lane in lanes.iter() {
            if let Some(index) = lane.index() {
                self.expiration_times[index as usize] = None;
            }
        }
    }

    /// Defers a retry that would replace one fallback with another.
    ///
    /// This is React's `finishConcurrentRender`, case `RootSuspended`, after
    /// the lanes were parked: a retry that shows a fallback waits until
    /// [`FALLBACK_THROTTLE_MS`] has passed since the last fallback that hid
    /// content, so a tree that retries through several levels does not flash a
    /// loading state for every level it reaches. Returns whether it took the
    /// render away from the caller, which happens when the tree is held for a
    /// later commit or thrown away for work that arrived meanwhile.
    ///
    /// React re-pings the still-suspended lanes instead of holding the tree
    /// when they name a level outside this render, so the next pass prefers the
    /// fallback of the last suspended level. Rea never reaches that case: a
    /// retry lane is only rendered after the update that claims it cleared the
    /// parked set, so the lanes this render parks are that whole set.
    fn defer_retry(&mut self, root: FiberId, lanes: Lanes, now: Option<u64>) -> bool {
        let Some(deadline) = now.and_then(|now| self.fallback_deadline(now)) else {
            return false;
        };
        // React's `getNextLanes(root, NoLanes) !== NoLanes`: additional work is
        // pending and replaces this tree, so holding it would only postpone the
        // commit that work performs.
        if !self.next_lanes().is_empty() {
            self.render = None;
            self.arena.abandon_render();
            self.interleaved = Lanes::NONE;
            return true;
        }
        // React's `scheduleTimeout(commitRoot, msUntilTimeout)` keeps
        // `root.finishedWork` alive until the handle fires. The fibers stay in
        // the arena, and the host's next slice commits them once the clock
        // reaches the deadline.
        self.hold(root, lanes, deadline);
        true
    }

    /// When the next fallback may commit, if it has to wait at all.
    ///
    /// This is React's `globalMostRecentFallbackTime + FALLBACK_THROTTLE_MS`
    /// measured against `now()`. A wait of a few milliseconds is not worth
    /// keeping the previous screen for — React's `msUntilTimeout > 10` — so a
    /// deadline that close answers `None` and the fallback commits at once.
    fn fallback_deadline(&self, now: u64) -> Option<u64> {
        let deadline = self
            .fallback_time?
            .saturating_add(FALLBACK_THROTTLE_MS);
        (deadline.saturating_sub(now) > MINIMUM_HOLD_MS).then_some(deadline)
    }

    /// Holds a finished tree until the next noticeable difference, if waiting
    /// is worth it.
    ///
    /// This is the rest of React's `finishConcurrentRender`, case
    /// `RootSuspendedWithDelay`: a render that would take visible content away
    /// waits for the next [`jnd`] boundary after the update it renders, because
    /// until then the user is unlikely to tell the wait from the placeholder it
    /// avoids. Returns whether it took the render away from the caller; `false`
    /// means the wait is too short to matter and the placeholder commits at
    /// once.
    ///
    /// A root without a clock cannot measure the wait, so it commits the
    /// placeholder immediately; work that arrives before the deadline replaces
    /// the held tree through [`Root::prepare_fresh`].
    fn defer_delay(&mut self, root: FiberId, lanes: Lanes, now: Option<u64>) -> bool {
        let Some(now) = now else {
            return false;
        };
        // React's `getMostRecentEventTime`: the wait is measured from the
        // newest update the render covers. A lane no clock has dated counts as
        // older than any reading — React's `NoTimestamp` — so the wait lands on
        // the next boundary above the current clock.
        let event_time = self.most_recent_event_time(lanes).unwrap_or(0);
        let elapsed = now.saturating_sub(event_time);
        let remaining = jnd(elapsed).saturating_sub(elapsed);
        if remaining <= MINIMUM_HOLD_MS {
            return false;
        }
        self.hold(root, lanes, now.saturating_add(remaining));
        true
    }

    /// Keeps the finished tree of a render until `deadline`.
    ///
    /// The render is over: its tree does not need to be resumed, only committed
    /// later, which is what React's `root.finishedWork` is for.
    fn hold(&mut self, root: FiberId, lanes: Lanes, deadline: u64) {
        self.render = None;
        self.held = Some(Held {
            root,
            lanes,
            deadline,
        });
    }

    /// The newest event time among `lanes`.
    ///
    /// This is React's `getMostRecentEventTime`: of the updates one render
    /// covers, the latest one says how long the user has been waiting, and a
    /// lane no clock has dated is left out.
    fn most_recent_event_time(&self, lanes: Lanes) -> Option<u64> {
        lanes
            .iter()
            .filter_map(|lane| {
                lane.index()
                    .and_then(|index| self.event_times[index as usize])
            })
            .max()
    }

    /// Dates `lanes` with the host's clock.
    ///
    /// This is the event time React records in `markRootUpdated`: a dispatch
    /// carries the moment it was requested, so a render that finishes later can
    /// tell how long the user has waited. A rea dispatch has no clock of its
    /// own — the host supplies one per slice — so the update is dated when the
    /// root drains it, and a lane that was marked without a clock keeps `None`
    /// until [`Root::mark_starved_lanes`] dates it.
    fn record_event_time(&mut self, lanes: Lanes, now: Option<u64>) {
        let Some(now) = now else {
            return;
        };
        for lane in lanes.iter() {
            if let Some(index) = lane.index() {
                self.event_times[index as usize] = Some(now);
            }
        }
    }

    /// Marks the parked lanes whose work resolved.
    ///
    /// This is React's `markRootPinged`. Only a lane that is still parked is
    /// marked: a lane that finished rendering no longer waits for the work, and
    /// the lanes stay suspended so [`Root::mark_suspended`] can park them again
    /// if the retry suspends in the same place.
    fn mark_root_pinged(&mut self, pinged: Lanes) {
        self.pinged_lanes
            .insert(self.suspended_lanes & pinged);
    }

    /// Schedules a render of the primary tree of a boundary whose work resolved.
    ///
    /// This is React's `retryTimedOutBoundary`: the boundary is marked with a
    /// fresh retry lane, exactly as an update would mark it, and the flush loop
    /// renders that lane like any other. The retry is allowed to suspend in the
    /// same place; the boundary then stays on its fallback until the new work
    /// resolves.
    ///
    /// The request names the fiber and the retry cache the listener was
    /// memoized in. A boundary that unmounted freed its slot, and the arena
    /// hands the slot out again, so the cache is what says the fiber is still
    /// the boundary that subscribed the listener; React checks the same thing
    /// with `getNearestMountedFiber`.
    fn retry_boundary(&mut self, request: RetryRequest, now: Option<u64>) {
        let mounted = self
            .arena
            .get(request.boundary)
            .is_some_and(|node| {
                node.retry_cache
                    .as_ref()
                    .is_some_and(|cache| Rc::ptr_eq(cache, &request.cache))
            });
        if !mounted {
            return;
        }
        let lane = self.claim_next_retry_lane();
        // React's `retryTimedOutBoundary` dates the retry like any other
        // update, with the moment it is requested.
        self.record_event_time(lane, now);
        self.mark_update(request.boundary, lane);
    }

    /// Takes the next retry lane, rotating through the four.
    ///
    /// This is React's `claimNextRetryLane`. The rotation is what keeps a new
    /// retry off a lane an earlier retry parked: a parked lane is skipped by
    /// [`Root::next_lanes`], so a retry that shared it would never render.
    fn claim_next_retry_lane(&mut self) -> Lanes {
        let lane = self.next_retry_lane;
        self.next_retry_lane = lane.next_retry();
        lane
    }

    /// Subscribes the root to the work a suspension waits for.
    ///
    /// This is React's `attachPingListener`. A lane can suspend on the same
    /// work more than once — the render retries and the component reads the
    /// same missing value — so the root keeps the pairs it already subscribed
    /// to and adds a listener once per pair. When the work reports back, the
    /// listener drops the wakeable: work that completed is never thrown again,
    /// and the retry reads what it produced.
    fn attach_ping_listener(&mut self, wakeable: &Wakeable, lanes: Lanes) {
        if self
            .ping_cache
            .borrow()
            .iter()
            .any(|(known, known_lanes)| known.same(wakeable) && *known_lanes == lanes)
        {
            return;
        }
        self.ping_cache
            .borrow_mut()
            .push((wakeable.clone(), lanes));
        let cache = Rc::clone(&self.ping_cache);
        let mailbox = Rc::clone(&self.mailbox);
        let known = wakeable.clone();
        wakeable.subscribe(Rc::new(move || {
            cache
                .borrow_mut()
                .retain(|(cached, _)| !cached.same(&known));
            mailbox.borrow_mut().ping(lanes);
        }));
    }

    /// Throws away a render in flight and starts a fresh tree for `lanes`.
    ///
    /// This is React's `prepareFreshStack`. The abandoned tree never reached
    /// the host: its fibers are freed, its update queues return to the state
    /// the last commit left, and the committed root is copied again. Work that
    /// arrived during the abandoned render is already in `pending_lanes`, so
    /// clearing the interleaved set here loses nothing. A held tree goes with
    /// it — React cancels the timeout a held fallback scheduled here — because
    /// the new render replaces the tree the hold would have committed.
    fn prepare_fresh(&mut self, lanes: Lanes) {
        self.held = None;
        self.arena.abandon_render();
        self.render = None;
        self.interleaved = Lanes::NONE;
        self.arena.begin_render();
        let Some(work) = self.arena.create_work_in_progress(self.current) else {
            self.arena.end_render();
            return;
        };
        let current_first = self
            .arena
            .get(self.current)
            .and_then(|fiber| fiber.first_child);
        if let Some(element) = self.element.as_ref() {
            reconcile_children(
                &self.host,
                &mut self.arena,
                work,
                current_first,
                slice::from_ref(element),
            );
        }
        self.render = Some(Render {
            root: work,
            next: Some(work),
            lanes,
            exit: ExitStatus::InProgress,
        });
    }

    /// Commits a finished tree and records the lanes the render left pending.
    ///
    /// `now` is the host's clock at the commit, if it supplied one.
    fn commit(&mut self, work: FiberId, lanes: Lanes, now: Option<u64>) {
        // `completeRoot`: the passive effects of the previous commit run before
        // this tree becomes current, so their destroy pass still visits the
        // fibers that created them.
        while self.run_passive() {}

        let (remaining, has_passive) = match self.arena.get(work) {
            Some(fiber) => (
                fiber
                    .lanes
                    .union(fiber.child_lanes)
                    .union(self.interleaved),
                fiber.flags.intersects(Flags::PASSIVE) || fiber.subtree_flags.intersects(Flags::PASSIVE),
            ),
            None => (self.interleaved, false),
        };
        // `markRootFinished`: a lane this render did not cover stays pending,
        // and so does the work that arrived while it rendered.
        let previously_pending = self.pending_lanes;
        self.pending_lanes = remaining;
        // The tree committed, so nothing below it is holding a lane back any
        // more; a boundary that still waits re-parks its retry lane when it
        // attaches one. React resets the same sets in `markRootFinished`.
        self.suspended_lanes = Lanes::NONE;
        self.pinged_lanes = Lanes::NONE;
        // A lane that is no longer pending starts over: a later update on it
        // gets a fresh expiration time instead of inheriting the one the
        // finished work had, and it is no longer expired. Its event time goes
        // with it, so a delayed fallback on that lane is not measured from an
        // update the previous screen already answered.
        for lane in (previously_pending & !remaining).iter() {
            if let Some(index) = lane.index() {
                let index = index as usize;
                self.expiration_times[index] = None;
                self.event_times[index] = None;
            }
        }
        self.expired_lanes &= remaining;

        let mut commit = Commit::new(&mut self.host, &mut self.arena, Rc::clone(&self.mailbox));
        commit.mutation(work);
        // React's `markCommitTimeOfFallback` runs inside the mutation phase;
        // the root only records the moment the commit reported so a later
        // fallback can be throttled against it. A commit with no clock leaves
        // the record alone: it cannot say when the fallback happened.
        if commit.hid_primary_tree()
            && let Some(now) = now
        {
            self.fallback_time = Some(now);
        }
        self.current = work;
        let mut commit = Commit::new(&mut self.host, &mut self.arena, Rc::clone(&self.mailbox));
        commit.layout(work);

        if has_passive {
            self.passive = Some(work);
        }

        // A commit that covers the sync-update group runs its passive effects
        // before it returns, so a discrete update is observable at the call
        // site rather than after the next frame.
        if lanes.intersects(Lanes::SYNC_UPDATE) {
            while self.run_passive() {}
        }
    }

    /// Runs the passive effects of the finished tree that is waiting for them.
    fn run_passive(&mut self) -> bool {
        let Some(root) = self.passive.take() else {
            return false;
        };
        let mut commit = Commit::new(&mut self.host, &mut self.arena, Rc::clone(&self.mailbox));
        commit.passive(root);
        true
    }

    /// Turns the pings, retries and dispatches collected since the last drain
    /// into pending lanes.
    ///
    /// A ping names the lanes whose work resolved; it is applied before the
    /// retries and the dispatches because a dispatch that clears the parked set
    /// would otherwise make the ping look as if it named nothing. A retry names
    /// a boundary whose caught work resolved; it marks the boundary the way an
    /// update would, so the flush loop renders the primary tree again. A queue
    /// whose fiber unmounted is detached and no longer wakes the root; should
    /// one still be signalled, its owner is `None` and the update is dropped,
    /// because a fiber that is not in the tree has no render to schedule. A
    /// dispatch that lands while a render is in flight is also recorded as
    /// interleaved: the render may be past the fiber it names, so its own lanes
    /// cannot be relied on to carry the lane to the completion.
    ///
    /// `now` is the host's clock, when this drain runs inside a slice. It dates
    /// the updates: a render that finishes later measures its wait from it.
    fn drain_mailbox(&mut self, now: Option<u64>) {
        let pinged = self.mailbox.borrow_mut().take_pinged();
        self.mark_root_pinged(pinged);
        let retries = self.mailbox.borrow_mut().drain_retries();
        for retry in retries {
            self.retry_boundary(retry, now);
        }
        let updates = self.mailbox.borrow_mut().drain();
        for (queue, lanes) in updates {
            let Some(owner) = queue.borrow().owner() else {
                continue;
            };
            if self.render.is_some() {
                self.interleaved.insert(lanes);
            }
            self.record_event_time(lanes, now);
            self.mark_update(owner, lanes);
        }
    }

    /// Marks `lanes` on `source`, on every ancestor and on the root.
    ///
    /// This is `markUpdateLaneFromFiberToRoot`: the source fiber records the
    /// lane so its render replays the update, each ancestor records it below
    /// itself so a bailout can tell there is work down there, and the root adds
    /// it to the pending set. The alternate of every fiber is marked too,
    /// because a dispatch can arrive while the tree that owns the queue is not
    /// the current one — a layout cleanup runs before the swap, a passive
    /// effect after the next render — and the lane must be visible on whichever
    /// half the next render starts from.
    fn mark_update(&mut self, source: FiberId, lanes: Lanes) {
        self.mark_lanes(source, lanes);
        let mut node = source;
        while let Some(parent) = self
            .arena
            .get(node)
            .and_then(|fiber| fiber.parent)
        {
            self.mark_child_lanes(parent, lanes);
            node = parent;
        }
        self.pending_lanes.insert(lanes);
        // `markRootUpdated`: a tree parked on a suspension is tried again when
        // any non-idle update arrives, because the update may be what the
        // parked work was waiting for. React clears the whole set rather than
        // the updated lanes, since it cannot tell which suspension an update
        // unblocks. Idle work never runs while regular work is pending, so it
        // cannot unblock anything and leaves the set alone.
        if !lanes.is_subset_of(Lanes::IDLE) {
            self.suspended_lanes = Lanes::NONE;
        }
    }

    /// Records `lanes` on a fiber and its alternate.
    fn mark_lanes(&mut self, fiber: FiberId, lanes: Lanes) {
        let alternate = self
            .arena
            .get(fiber)
            .and_then(|fiber| fiber.alternate);
        for id in core::iter::once(fiber).chain(alternate) {
            if let Some(node) = self.arena.get_mut(id) {
                node.lanes.insert(lanes);
            }
        }
    }

    /// Records `lanes` below a fiber, on it and its alternate.
    fn mark_child_lanes(&mut self, fiber: FiberId, lanes: Lanes) {
        let alternate = self
            .arena
            .get(fiber)
            .and_then(|fiber| fiber.alternate);
        for id in core::iter::once(fiber).chain(alternate) {
            if let Some(node) = self.arena.get_mut(id) {
                node.child_lanes.insert(lanes);
            }
        }
    }
}

/// How long after a fallback that hid content the next one may commit.
///
/// This is React's `FALLBACK_THROTTLE_MS`.
const FALLBACK_THROTTLE_MS: u64 = 500;

/// The shortest wait worth holding a finished tree for.
///
/// This is React's `msUntilTimeout > 10`: replacing the screen a few
/// milliseconds later is not worth keeping the previous one.
const MINIMUM_HOLD_MS: u64 = 10;

/// The next Just Noticeable Difference boundary after `time_elapsed`.
///
/// This is React's `jnd`. A person cannot tell small differences in time apart,
/// so waiting a little longer than necessary costs nothing while it may spare
/// an intermediate loading state. The longer the wait already was, the longer
/// an added wait is imperceptible, so the boundaries widen on a fixed schedule
/// up to a point and then repeat at a constant interval.
fn jnd(time_elapsed: u64) -> u64 {
    match time_elapsed {
        0..120 => 120,
        120..480 => 480,
        480..1080 => 1080,
        1080..1920 => 1920,
        1920..3000 => 3000,
        3000..4320 => 4320,
        _ => time_elapsed.div_ceil(1960) * 1960,
    }
}

/// Whether `lanes` hold nothing but retries.
///
/// This is React's `includesOnlyRetries`: `(lanes & RetryLanes) === lanes`. A
/// retry renders no user update, so a fallback it shows is the only thing that
/// changed on screen and React throttles it rather than flashing it.
fn includes_only_retries(lanes: Lanes) -> bool {
    lanes & Lanes::RETRY == lanes
}

/// Whether `lanes` hold nothing but transition updates.
///
/// This is React's `includesOnlyTransitions`: `(lanes & TransitionLanes) ===
/// lanes`. Only the ten update lanes count — a deferred value of a transition
/// is not a transition of its own — and the caller passes render lanes, which
/// are never empty.
fn includes_only_transitions(lanes: Lanes) -> bool {
    lanes & Lanes::TRANSITION_UPDATE_LANES == lanes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Reference;
    use crate::hooks::{Cleanup, State};
    use crate::wakeable::Wakeable;
    use alloc::rc::Rc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::{Cell, RefCell};

    /// The handle of a node in [`TreeHost`].
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct NodeId(u32);

    /// One node: its element, the children the commit attached, how many
    /// times the host updated it and whether it is shown.
    struct NodeState {
        element: Option<u32>,
        children: Vec<NodeId>,
        updates: u32,
        hidden: bool,
    }

    /// A host that keeps the child order, so a placement can be asserted.
    struct TreeHost {
        nodes: Vec<NodeState>,
    }

    impl TreeHost {
        fn new() -> Self {
            Self {
                nodes: vec![NodeState {
                    element: None,
                    children: Vec::new(),
                    updates: 0,
                    hidden: false,
                }],
            }
        }

        fn container(&self) -> NodeId {
            NodeId(0)
        }

        fn children(&self, node: NodeId) -> &[NodeId] {
            &self.nodes[node.0 as usize].children
        }

        fn element(&self, node: NodeId) -> Option<u32> {
            self.nodes[node.0 as usize].element
        }

        fn hidden(&self, node: NodeId) -> bool {
            self.nodes[node.0 as usize].hidden
        }
    }

    impl Host for TreeHost {
        type Element = u32;
        type Node = NodeId;

        fn element_matches(&self, old: &u32, new: &u32) -> bool {
            old == new
        }

        fn create(&mut self, element: &u32) -> NodeId {
            let id = NodeId(self.nodes.len() as u32);
            self.nodes.push(NodeState {
                element: Some(*element),
                children: Vec::new(),
                updates: 0,
                hidden: false,
            });
            id
        }

        fn update(&mut self, node: NodeId, element: &u32) {
            if let Some(state) = self.nodes.get_mut(node.0 as usize) {
                state.element = Some(*element);
                state.updates += 1;
            }
        }

        fn insert_before(&mut self, parent: NodeId, child: NodeId, anchor: Option<NodeId>) {
            for state in &mut self.nodes {
                state.children.retain(|current| *current != child);
            }
            let Some(state) = self.nodes.get_mut(parent.0 as usize) else {
                return;
            };
            match anchor.and_then(|anchor| {
                state
                    .children
                    .iter()
                    .position(|node| *node == anchor)
            }) {
                Some(index) => state.children.insert(index, child),
                None => state.children.push(child),
            }
        }

        fn remove(&mut self, parent: NodeId, child: NodeId) {
            if let Some(state) = self.nodes.get_mut(parent.0 as usize) {
                state.children.retain(|current| *current != child);
            }
        }

        fn hide(&mut self, node: NodeId) {
            if let Some(state) = self.nodes.get_mut(node.0 as usize) {
                state.hidden = true;
            }
        }

        fn unhide(&mut self, node: NodeId) {
            if let Some(state) = self.nodes.get_mut(node.0 as usize) {
                state.hidden = false;
            }
        }
    }

    /// Creates a root with `element` scheduled but not rendered.
    fn scheduled(element: Element<TreeHost>) -> Root<TreeHost> {
        let host = TreeHost::new();
        let container = host.container();
        let mut root = Root::new(host, container);
        root.render(element);
        root
    }

    /// Mounts `element` and leaves the root with no pending work.
    fn mount(element: Element<TreeHost>) -> Root<TreeHost> {
        let mut root = scheduled(element);
        root.flush();
        root
    }

    /// The props of the container's children, in order.
    fn top_level(root: &Root<TreeHost>) -> Vec<Option<u32>> {
        root.host()
            .children(root.host().container())
            .iter()
            .map(|node| root.host().element(*node))
            .collect()
    }

    /// A component that renders its state, records every rendered value and
    /// hands its state handle to the test.
    fn counter(handle: &Rc<RefCell<Option<State<u32>>>>, log: &Rc<RefCell<Vec<u32>>>) -> Element<TreeHost> {
        let handle = Rc::clone(handle);
        let log = Rc::clone(log);
        Element::component(move |ctx| {
            let value = ctx.use_state(|| 1_u32);
            *handle.borrow_mut() = Some(value.clone());
            log.borrow_mut().push(*value.get());
            Element::host(*value.get())
        })
    }

    /// The two state handles a [`pair`] component hands to its test.
    type PairHandles = Rc<RefCell<Option<(State<u32>, State<u32>)>>>;

    /// A component with two states: it renders the first one, records both
    /// values on every render and hands the handles to the test.
    fn pair(handles: &PairHandles, log: &Rc<RefCell<Vec<(u32, u32)>>>) -> Element<TreeHost> {
        let handles = Rc::clone(handles);
        let log = Rc::clone(log);
        Element::component(move |ctx| {
            let first = ctx.use_state(|| 1_u32);
            let second = ctx.use_state(|| 1_u32);
            *handles.borrow_mut() = Some((first.clone(), second.clone()));
            log.borrow_mut()
                .push((*first.get(), *second.get()));
            Element::host(*first.get())
        })
    }

    /// A component that renders the deferred value of its state, records what
    /// it shows and hands the state handle to the test.
    fn deferred(handle: &Rc<RefCell<Option<State<u32>>>>, log: &Rc<RefCell<Vec<u32>>>) -> Element<TreeHost> {
        let handle = Rc::clone(handle);
        let log = Rc::clone(log);
        Element::component(move |ctx| {
            let value = ctx.use_state(|| 1_u32);
            *handle.borrow_mut() = Some(value.clone());
            let shown = ctx.use_deferred_value(*value.get());
            log.borrow_mut().push(shown);
            Element::host(shown)
        })
    }

    #[test]
    fn a_deferred_value_shows_the_old_value_until_its_lane_renders() {
        let handle = Rc::new(RefCell::new(None));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(deferred(&handle, &log));
        let Some(value) = handle.borrow().clone() else {
            return;
        };

        value.set(2);
        root.flush();

        // The urgent render keeps showing the old value and leaves the deferred
        // lane pending; the root then renders that lane, which catches up.
        assert_eq!(*log.borrow(), [1, 1, 2]);
        assert_eq!(top_level(&root), [Some(2)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_transition_update_waits_for_the_default_lane() {
        let handles = Rc::new(RefCell::new(None));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(pair(&handles, &log));
        let Some((first, second)) = handles.borrow().clone() else {
            return;
        };

        root.start_transition(|| first.set(2));
        second.set(3);
        root.flush();

        // The default lane renders first and sees the transition's update still
        // pending; the transition lane then applies it.
        assert_eq!(*log.borrow(), [(1, 1), (1, 3), (2, 3)]);
        assert_eq!(top_level(&root), [Some(2)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn transitions_started_before_a_flush_share_a_lane() {
        let handles = Rc::new(RefCell::new(None));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(pair(&handles, &log));
        let Some((first, second)) = handles.borrow().clone() else {
            return;
        };

        root.start_transition(|| first.set(2));
        root.start_transition(|| second.set(3));
        root.flush();

        // Both scopes ran in the same event, so their updates carry one lane
        // and reach the host in a single render.
        assert_eq!(*log.borrow(), [(1, 1), (2, 3)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_mount_attaches_the_element_to_the_container() {
        let mut root = mount(Element::fragment([
            Element::host(1),
            Element::host(2),
            Element::host(3),
        ]));
        assert_eq!(top_level(&root), [Some(1), Some(2), Some(3)]);
        assert!(!root.has_pending_work());

        // Nothing is pending, so a second flush touches nothing.
        let created = root.host().nodes.len();
        root.flush();
        assert_eq!(root.host().nodes.len(), created);
    }

    #[test]
    fn an_empty_element_clears_the_container() {
        let mut root = mount(Element::host(1));
        assert_eq!(top_level(&root), [Some(1)]);
        root.render(Element::empty());
        root.flush();
        assert!(top_level(&root).is_empty());
    }

    /// A component that renders one host node until its state says its data is
    /// missing, and suspends on a wakeable nothing resolves until then.
    fn sometimes(handle: &Rc<RefCell<Option<State<bool>>>>, rendered: &Rc<Cell<u32>>) -> Element<TreeHost> {
        let handle = Rc::clone(handle);
        let rendered = Rc::clone(rendered);
        Element::component(move |ctx| {
            let missing = ctx.use_state(|| false);
            *handle.borrow_mut() = Some(missing.clone());
            rendered.set(rendered.get() + 1);
            if *missing.get() {
                return ctx.suspend(Wakeable::from_fn(|_| {}));
            }
            Element::host(7)
        })
    }

    #[test]
    fn a_suspension_at_the_shell_keeps_the_committed_tree() {
        let handle = Rc::new(RefCell::new(None));
        let rendered = Rc::new(Cell::new(0));
        let mut root = mount(sometimes(&handle, &rendered));
        assert_eq!(top_level(&root), [Some(7)]);
        let nodes = root.host().nodes.len();
        let fibers = root.arena.len();
        let Some(missing) = handle.borrow().clone() else {
            return;
        };

        missing.set(true);
        root.flush();

        // The component rendered again and suspended, so the pass stopped
        // there: the element it never built reached no host, and the fibers of
        // the thrown-away tree are gone.
        assert_eq!(rendered.get(), 2);
        assert_eq!(top_level(&root), [Some(7)]);
        assert_eq!(root.host().nodes.len(), nodes);
        assert_eq!(root.arena.len(), fibers);
        // The lane stays pending, but parked: no flush can progress on it.
        assert_eq!(root.pending_lanes, Lanes::DEFAULT);
        assert_eq!(root.suspended_lanes, Lanes::DEFAULT);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn an_update_after_a_suspension_tries_the_parked_lane_again() {
        let handle = Rc::new(RefCell::new(None));
        let rendered = Rc::new(Cell::new(0));
        let mut root = mount(sometimes(&handle, &rendered));
        let nodes = root.host().nodes.len();
        let Some(missing) = handle.borrow().clone() else {
            return;
        };
        missing.set(true);
        root.flush();
        assert!(!root.has_pending_work());

        // The handle the committed render handed out still reaches its queue,
        // and the update is what unblocks the parked lane.
        missing.set(false);
        root.flush();

        assert_eq!(rendered.get(), 3);
        assert_eq!(top_level(&root), [Some(7)]);
        assert_eq!(root.suspended_lanes, Lanes::NONE);
        assert!(!root.has_pending_work());
        assert_eq!(root.host().nodes.len(), nodes);
    }

    /// The listeners a [`manual`] wakeable has registered, in order.
    type Listeners = Rc<RefCell<Vec<Rc<dyn Fn()>>>>;

    /// A wakeable the test resolves by hand.
    ///
    /// The subscribers stay in the returned list so the test can fire the work
    /// the way a completed request would; the engine's listener is one of them.
    fn manual() -> (Wakeable, Listeners) {
        let subscribers: Listeners = Rc::new(RefCell::new(Vec::new()));
        let waiting = Rc::clone(&subscribers);
        let wakeable = Wakeable::from_fn(move |wake| waiting.borrow_mut().push(wake));
        (wakeable, subscribers)
    }

    /// Runs every listener `work` registered.
    fn resolve(subscribers: &Listeners) {
        for wake in subscribers.borrow().iter() {
            wake();
        }
    }

    /// Runs the listener `work` registered last, which is the one the latest
    /// capture subscribed.
    ///
    /// A wakeable calls its subscribers once, when it completes; a test that
    /// resumes the same work twice must not re-run the listener of the
    /// resolution that already happened.
    fn resolve_latest(subscribers: &Listeners) {
        if let Some(wake) = subscribers.borrow().last() {
            wake();
        }
    }

    /// A component that renders one host node once `ready` is set and suspends
    /// on `work` until then.
    fn waiting_on(ready: &Rc<Cell<bool>>, work: &Wakeable) -> Element<TreeHost> {
        let ready = Rc::clone(ready);
        let work = work.clone();
        Element::component(move |ctx| {
            if ready.get() {
                Element::host(7)
            } else {
                ctx.suspend(work.clone())
            }
        })
    }

    #[test]
    fn a_shell_suspension_retries_when_its_work_resolves() {
        let ready = Rc::new(Cell::new(false));
        let (work, subscribers) = manual();
        let mut root = mount(waiting_on(&ready, &work));

        // The suspension parked the lane and left the container empty.
        assert!(top_level(&root).is_empty());
        assert_eq!(root.suspended_lanes, Lanes::DEFAULT);
        assert!(!root.has_pending_work());

        // The work resolved while the tree was parked: the ping is what makes
        // the lane a candidate again, and the host sees it before flushing.
        ready.set(true);
        resolve(&subscribers);
        assert!(root.has_pending_work());
        root.flush();

        assert_eq!(top_level(&root), [Some(7)]);
        assert_eq!(root.suspended_lanes, Lanes::NONE);
        assert_eq!(root.pinged_lanes, Lanes::NONE);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn the_root_subscribes_a_wakeable_once_per_lane() {
        let ready = Rc::new(Cell::new(false));
        let (work, subscribers) = manual();
        let mut root = mount(waiting_on(&ready, &work));
        assert_eq!(subscribers.borrow().len(), 1);

        // The retry suspends on the same work at the same lane. The wakeable is
        // already subscribed for that lane, so the root adds no second listener
        // and the lane parks again.
        root.render(waiting_on(&ready, &work));
        root.flush();
        assert_eq!(root.suspended_lanes, Lanes::DEFAULT);
        assert_eq!(subscribers.borrow().len(), 1);
    }

    #[test]
    fn a_boundary_retries_its_primary_tree_when_its_work_resolves() {
        let ready = Rc::new(Cell::new(false));
        let (work, subscribers) = manual();
        let element = || Element::suspense(waiting_on(&ready, &work), Element::host(9));
        let mut root = mount(element());

        // The boundary caught the suspension: the host shows its fallback, and
        // the commit subscribed the work the primary tree waits on.
        assert_eq!(top_level(&root), [Some(9)]);
        assert_eq!(subscribers.borrow().len(), 1);
        assert_eq!(root.suspended_lanes, Lanes::NONE);
        assert!(!root.has_pending_work());

        // The work resolved. The listener marks the boundary with a retry lane,
        // which is what tells the host the tree has something to try, before
        // anything is flushed.
        ready.set(true);
        resolve(&subscribers);
        assert!(root.has_pending_work());
        root.flush();

        // The boundary reconciled its primary tree again: the primary reached
        // the host and the fallback was removed.
        assert_eq!(top_level(&root), [Some(7)]);
        assert!(root.pending_lanes.is_empty());
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_boundary_retry_that_suspends_again_waits_for_the_new_work() {
        let ready = Rc::new(Cell::new(false));
        let (work, subscribers) = manual();
        let element = || Element::suspense(waiting_on(&ready, &work), Element::host(9));
        let mut root = mount(element());
        assert_eq!(top_level(&root), [Some(9)]);

        // The work resolved but the data is still missing: the retry suspends
        // in the same place, the boundary keeps its fallback, and the work is
        // subscribed again because the cache forgot it when it resolved.
        resolve_latest(&subscribers);
        root.flush();
        assert_eq!(top_level(&root), [Some(9)]);
        assert_eq!(subscribers.borrow().len(), 2);
        assert!(root.pending_lanes.is_empty());
        assert!(!root.has_pending_work());

        // The second resolution is the one that brings the primary tree back.
        ready.set(true);
        resolve_latest(&subscribers);
        root.flush();
        assert_eq!(top_level(&root), [Some(7)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_hidden_tree_unmounts_its_layout_effects_until_it_reappears() {
        let handle = Rc::new(RefCell::new(None::<State<bool>>));
        let ready = Rc::new(Cell::new(false));
        let log: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
        let (work, subscribers) = manual();
        let element = {
            let handle = Rc::clone(&handle);
            let ready = Rc::clone(&ready);
            let log = Rc::clone(&log);
            let work = work.clone();
            Element::suspense(
                Element::component(move |ctx| {
                    let missing = ctx.use_state(|| false);
                    *handle.borrow_mut() = Some(missing.clone());
                    let created = Rc::clone(&log);
                    let cleaned = Rc::clone(&log);
                    // The dependencies never change, so neither render of the
                    // returning tree queues an effect: only the reappearance
                    // brings the effect back.
                    ctx.use_layout_effect((), move || {
                        created.borrow_mut().push("create");
                        let cleaned = Rc::clone(&cleaned);
                        Some(Box::new(move || cleaned.borrow_mut().push("cleanup")) as Cleanup)
                    });
                    if *missing.get() && !ready.get() {
                        ctx.suspend(work.clone())
                    } else {
                        Element::host(7)
                    }
                }),
                Element::host(9),
            )
        };
        let mut root = mount(element);
        assert_eq!(top_level(&root), [Some(7)]);
        assert_eq!(*log.borrow(), ["create"]);

        // The update suspends: the boundary puts up its fallback and the commit
        // takes the hidden tree's layout cleanup. The suspended render queued
        // no effect of its own: its fibers were thrown away.
        let Some(missing) = handle.borrow().clone() else {
            return;
        };
        missing.set(true);
        root.flush();
        assert_eq!(top_level(&root), [Some(7), Some(9)]);
        let primary = mounted(&root)[0];
        assert!(root.host().hidden(primary));
        assert_eq!(*log.borrow(), ["create", "cleanup"]);

        // The work resolved: the retry shows the primary tree again. The retry
        // lane does not cover the component's update, so the component itself is
        // not re-rendered and the reappearance is what mounts its layout effect,
        // from the record the hooks kept.
        ready.set(true);
        resolve(&subscribers);
        root.flush();
        assert_eq!(top_level(&root), [Some(7)]);
        assert!(!root.host().hidden(primary));
        assert_eq!(*log.borrow(), ["create", "cleanup", "create"]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_hidden_tree_detaches_its_references_until_it_reappears() {
        let handle = Rc::new(RefCell::new(None::<State<bool>>));
        let ready = Rc::new(Cell::new(false));
        let calls = Rc::new(RefCell::new(Vec::new()));
        let reference: Reference<TreeHost> = {
            let calls = Rc::clone(&calls);
            Rc::new(move |node| {
                calls.borrow_mut().push(node);
                None
            })
        };
        let (work, subscribers) = manual();
        let element = {
            let handle = Rc::clone(&handle);
            let ready = Rc::clone(&ready);
            let reference = Rc::clone(&reference);
            let work = work.clone();
            Element::suspense(
                Element::component(move |ctx| {
                    let missing = ctx.use_state(|| false);
                    *handle.borrow_mut() = Some(missing.clone());
                    if *missing.get() && !ready.get() {
                        ctx.suspend(work.clone())
                    } else {
                        Element::host(7).referenced(Rc::clone(&reference))
                    }
                }),
                Element::host(9),
            )
        };
        let mut root = mount(element);
        let node = mounted(&root)[0];
        assert_eq!(*calls.borrow(), [Some(node)]);

        // A hidden tree is not on screen, so its reference lets go of the node
        // even though the node stays mounted.
        let Some(missing) = handle.borrow().clone() else {
            return;
        };
        missing.set(true);
        root.flush();
        assert_eq!(top_level(&root), [Some(7), Some(9)]);
        assert!(root.host().hidden(node));
        assert_eq!(*calls.borrow(), [Some(node), None]);

        // The reappearing tree attaches the node again. The reference did not
        // change, so no render marked the fiber: only the reappearance can.
        ready.set(true);
        resolve(&subscribers);
        root.flush();
        assert_eq!(top_level(&root), [Some(7)]);
        assert!(!root.host().hidden(node));
        assert_eq!(*calls.borrow(), [Some(node), None, Some(node)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_hidden_boundary_leaves_an_already_hidden_tree_alone() {
        let inner_handle = Rc::new(RefCell::new(None::<State<bool>>));
        let outer_handle = Rc::new(RefCell::new(None::<State<bool>>));
        let inner_ready = Rc::new(Cell::new(false));
        let outer_ready = Rc::new(Cell::new(false));
        let log: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
        let (inner_work, inner_subscribers) = manual();
        let (outer_work, outer_subscribers) = manual();
        let element = {
            let inner_handle = Rc::clone(&inner_handle);
            let outer_handle = Rc::clone(&outer_handle);
            let inner_ready = Rc::clone(&inner_ready);
            let outer_ready = Rc::clone(&outer_ready);
            let log = Rc::clone(&log);
            let inner_work = inner_work.clone();
            let outer_work = outer_work.clone();
            let inner = Element::suspense(
                Element::component(move |ctx| {
                    let missing = ctx.use_state(|| false);
                    *inner_handle.borrow_mut() = Some(missing.clone());
                    let created = Rc::clone(&log);
                    let cleaned = Rc::clone(&log);
                    ctx.use_layout_effect((), move || {
                        created.borrow_mut().push("create");
                        let cleaned = Rc::clone(&cleaned);
                        Some(Box::new(move || cleaned.borrow_mut().push("cleanup")) as Cleanup)
                    });
                    if *missing.get() && !inner_ready.get() {
                        ctx.suspend(inner_work.clone())
                    } else {
                        Element::host(7)
                    }
                }),
                Element::host(8),
            );
            let outer_child = Element::component(move |ctx| {
                let missing = ctx.use_state(|| false);
                *outer_handle.borrow_mut() = Some(missing.clone());
                if *missing.get() && !outer_ready.get() {
                    ctx.suspend(outer_work.clone())
                } else {
                    Element::host(6)
                }
            });
            Element::suspense(Element::fragment([inner, outer_child]), Element::host(9))
        };
        let mut root = mount(element);
        assert_eq!(top_level(&root), [Some(7), Some(6)]);
        let seven = mounted(&root)[0];
        assert_eq!(*log.borrow(), ["create"]);

        // The inner boundary catches its own suspension: its primary tree goes
        // out of sight and the host shows the inner fallback.
        let Some(inner_missing) = inner_handle.borrow().clone() else {
            return;
        };
        inner_missing.set(true);
        root.flush();
        assert_eq!(top_level(&root), [Some(7), Some(8), Some(6)]);
        let eight = mounted(&root)[1];
        assert!(root.host().hidden(seven));
        assert!(!root.host().hidden(eight));
        assert_eq!(*log.borrow(), ["create", "cleanup"]);

        // The outer boundary now catches a suspension of its own. Its primary
        // tree contains the inner boundary, so the disappearing walk meets a
        // tree the inner boundary already hid: that tree must not lose its
        // effects a second time.
        let Some(outer_missing) = outer_handle.borrow().clone() else {
            return;
        };
        outer_missing.set(true);
        root.flush();
        assert_eq!(top_level(&root), [Some(7), Some(8), Some(6), Some(9)]);
        assert!(root.host().hidden(eight));
        assert!(root.host().hidden(seven));
        assert_eq!(*log.borrow(), ["create", "cleanup"]);

        // The outer work resolved: the reappearing walk shows its own subtree
        // but stops at the inner hidden tree, which keeps waiting for its work.
        outer_ready.set(true);
        resolve(&outer_subscribers);
        root.flush();
        assert_eq!(top_level(&root), [Some(7), Some(8), Some(6)]);
        assert!(!root.host().hidden(eight));
        assert!(root.host().hidden(seven));
        assert_eq!(*log.borrow(), ["create", "cleanup"]);

        // The inner work resolved last, so the inner tree is the one that comes
        // back, and it creates the effect exactly once.
        inner_ready.set(true);
        resolve(&inner_subscribers);
        root.flush();
        assert_eq!(top_level(&root), [Some(7), Some(6)]);
        assert!(!root.host().hidden(seven));
        assert_eq!(*log.borrow(), ["create", "cleanup", "create"]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_transition_that_has_to_hide_content_keeps_the_previous_screen() {
        let handle = Rc::new(RefCell::new(None::<State<bool>>));
        let ready = Rc::new(Cell::new(false));
        let renders = Rc::new(Cell::new(0));
        let (work, subscribers) = manual();
        let element = {
            let handle = Rc::clone(&handle);
            let ready = Rc::clone(&ready);
            let renders = Rc::clone(&renders);
            let work = work.clone();
            Element::suspense(
                Element::component(move |ctx| {
                    let missing = ctx.use_state(|| false);
                    *handle.borrow_mut() = Some(missing.clone());
                    renders.set(renders.get() + 1);
                    let missing = *missing.get();
                    if missing && !ready.get() {
                        return ctx.suspend(work.clone());
                    }
                    Element::host(if missing { 8 } else { 7 })
                }),
                Element::host(9),
            )
        };
        let mut root = mount(element);
        assert_eq!(top_level(&root), [Some(7)]);
        let Some(missing) = handle.borrow().clone() else {
            return;
        };

        // The transition needs data the primary tree does not have, and showing
        // the fallback would take away the content on screen. A transition may
        // wait, so nothing commits and the lane parks.
        missing.set_with_lane(Lanes::TRANSITION.highest(), true);
        root.flush();

        assert_eq!(renders.get(), 2);
        assert_eq!(top_level(&root), [Some(7)]);
        assert_eq!(root.suspended_lanes, Lanes::TRANSITION.highest());
        assert!(!root.has_pending_work());
        // The render that would have shown the fallback is gone, so the work it
        // waited on is subscribed by the root instead of by a boundary.
        assert_eq!(subscribers.borrow().len(), 1);

        // The work resolved: the ping makes the parked lane a candidate again
        // and the transition finds the data, so it commits.
        ready.set(true);
        resolve(&subscribers);
        assert!(root.has_pending_work());
        root.flush();

        assert_eq!(top_level(&root), [Some(8)]);
        assert_eq!(root.suspended_lanes, Lanes::NONE);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_transition_that_mounts_a_suspending_boundary_shows_its_fallback() {
        let handle = Rc::new(RefCell::new(None::<State<bool>>));
        let ready = Rc::new(Cell::new(false));
        let (work, _subscribers) = manual();
        let element = {
            let handle = Rc::clone(&handle);
            let ready = Rc::clone(&ready);
            let work = work.clone();
            Element::component(move |ctx| {
                let show = ctx.use_state(|| false);
                *handle.borrow_mut() = Some(show.clone());
                if *show.get() {
                    Element::suspense(waiting_on(&ready, &work), Element::host(9))
                } else {
                    Element::host(1)
                }
            })
        };
        let mut root = mount(element);
        assert_eq!(top_level(&root), [Some(1)]);
        let Some(show) = handle.borrow().clone() else {
            return;
        };

        // The boundary does not exist yet, so its fallback hides nothing: a
        // transition commits the initial loading state instead of waiting for
        // data that may never be needed.
        show.set_with_lane(Lanes::TRANSITION.highest(), true);
        root.flush();

        assert_eq!(top_level(&root), [Some(9)]);
        assert_eq!(root.suspended_lanes, Lanes::NONE);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_fallback_inside_a_hidden_tree_commits_during_a_transition() {
        let inner_handle = Rc::new(RefCell::new(None::<State<bool>>));
        let outer_handle = Rc::new(RefCell::new(None::<State<bool>>));
        let (inner_work, _inner_subscribers) = manual();
        let (outer_work, _outer_subscribers) = manual();
        let element = {
            let inner_handle = Rc::clone(&inner_handle);
            let outer_handle = Rc::clone(&outer_handle);
            let inner_work = inner_work.clone();
            let outer_work = outer_work.clone();
            let inner = Element::suspense(
                Element::component(move |ctx| {
                    let missing = ctx.use_state(|| false);
                    *inner_handle.borrow_mut() = Some(missing.clone());
                    if *missing.get() {
                        ctx.suspend(inner_work.clone())
                    } else {
                        Element::host(5)
                    }
                }),
                Element::host(6),
            );
            let outer_child = Element::component(move |ctx| {
                let missing = ctx.use_state(|| false);
                *outer_handle.borrow_mut() = Some(missing.clone());
                if *missing.get() {
                    ctx.suspend(outer_work.clone())
                } else {
                    Element::host(4)
                }
            });
            Element::suspense(Element::fragment([inner, outer_child]), Element::host(9))
        };
        let mut root = mount(element);
        assert_eq!(top_level(&root), [Some(5), Some(4)]);

        // The outer boundary hides its whole primary tree, the inner boundary
        // included, behind its fallback. The update that made it suspend went
        // with the primary attempt the boundary threw away, so the hidden tree's
        // lanes do not stay pending: the fallback commit covers them.
        let Some(outer_missing) = outer_handle.borrow().clone() else {
            return;
        };
        outer_missing.set(true);
        root.flush();
        assert_eq!(top_level(&root), [Some(5), Some(4), Some(9)]);
        assert!(!root.has_pending_work());

        // The transition reaches the inner boundary through the hidden tree and
        // makes it show its fallback. That fallback hides nothing the user can
        // see — the tree around it is already out of sight — so the boundary
        // does not ask the render to wait, and the nested fallback is committed
        // instead of the transition lane parking.
        let Some(inner_missing) = inner_handle.borrow().clone() else {
            return;
        };
        inner_missing.set_with_lane(Lanes::TRANSITION.highest(), true);
        root.flush();

        // Revealing the primary tree hands the outer child's own update back to
        // the root, which the transition render bailed out of; applying it hides
        // the tree once more. The nested fallback is inside it: the transition
        // committed rather than being thrown away.
        assert_eq!(top_level(&root), [Some(5), Some(6), Some(4), Some(9)]);
        let mounted = mounted(&root);
        assert!(root.host().hidden(mounted[0]));
        assert!(root.host().hidden(mounted[1]));
        assert!(root.host().hidden(mounted[2]));
        assert!(!root.host().hidden(mounted[3]));
        assert_eq!(root.suspended_lanes, Lanes::NONE);
        assert!(!root.has_pending_work());
    }

    /// An element whose boundary suspends on `work` until `ready` and then
    /// reveals a nested boundary whose content suspends on `nested`.
    fn revealing_nested(ready: &Rc<Cell<bool>>, work: &Wakeable, nested: &Wakeable) -> Element<TreeHost> {
        let ready = Rc::clone(ready);
        let work = work.clone();
        let nested = nested.clone();
        let missing = Rc::new(Cell::new(false));
        Element::suspense(
            Element::component(move |ctx| {
                if ready.get() {
                    Element::suspense(waiting_on(&missing, &nested), Element::host(8))
                } else {
                    ctx.suspend(work.clone())
                }
            }),
            Element::host(9),
        )
    }

    #[test]
    fn a_retry_that_reveals_a_new_fallback_waits_for_the_throttle() {
        let ready = Rc::new(Cell::new(false));
        let (work, subscribers) = manual();
        let (nested, _nested_subscribers) = manual();
        let host = TreeHost::new();
        let container = host.container();
        let mut root = Root::new(host, container);
        root.render(revealing_nested(&ready, &work, &nested));

        // The mount shows the first fallback at t=0. The commit hid a primary
        // tree, so the throttle starts here even though nothing was on screen.
        assert!(!root.flush_slice(0, || false));
        assert_eq!(top_level(&root), [Some(9)]);

        // The outer work resolved, and the retry mounts a boundary that
        // suspends at once. Its fallback would replace the one committed
        // 100 ms ago, so the finished tree is held and its retry lane parked.
        ready.set(true);
        resolve(&subscribers);
        assert!(root.flush_slice(100, || false));
        assert_eq!(top_level(&root), [Some(9)]);
        assert!(root.has_pending_work());

        // Nothing commits while the throttle is still running.
        assert!(root.flush_slice(499, || false));
        assert_eq!(top_level(&root), [Some(9)]);

        // At the deadline the held tree commits and the nested fallback takes
        // the place of the one it replaced.
        assert!(!root.flush_slice(500, || false));
        assert_eq!(top_level(&root), [Some(8)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_retry_after_the_throttle_commits_its_fallback_at_once() {
        let ready = Rc::new(Cell::new(false));
        let (work, subscribers) = manual();
        let (nested, _nested_subscribers) = manual();
        let host = TreeHost::new();
        let container = host.container();
        let mut root = Root::new(host, container);
        root.render(revealing_nested(&ready, &work, &nested));
        assert!(!root.flush_slice(0, || false));

        // The next fallback comes a full throttle later, so there is nothing
        // left to wait for and the commit happens in this slice.
        ready.set(true);
        resolve(&subscribers);
        assert!(!root.flush_slice(500, || false));
        assert_eq!(top_level(&root), [Some(8)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_flush_without_a_clock_commits_a_throttled_fallback_at_once() {
        let ready = Rc::new(Cell::new(false));
        let (work, subscribers) = manual();
        let (nested, _nested_subscribers) = manual();
        let host = TreeHost::new();
        let container = host.container();
        let mut root = Root::new(host, container);
        root.render(revealing_nested(&ready, &work, &nested));
        assert!(!root.flush_slice(0, || false));

        // `flush` supplies no clock, so it cannot schedule the wait and the
        // fallback commits at once instead of being held.
        ready.set(true);
        resolve(&subscribers);
        root.flush();
        assert_eq!(top_level(&root), [Some(8)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_delayed_fallback_waits_for_the_next_noticeable_difference() {
        let handle = Rc::new(RefCell::new(None::<State<bool>>));
        let ready = Rc::new(Cell::new(false));
        let (work, _subscribers) = manual();
        let element = {
            let handle = Rc::clone(&handle);
            let ready = Rc::clone(&ready);
            let work = work.clone();
            Element::suspense(
                Element::component(move |ctx| {
                    let missing = ctx.use_state(|| false);
                    *handle.borrow_mut() = Some(missing.clone());
                    if *missing.get() && !ready.get() {
                        return ctx.suspend(work.clone());
                    }
                    Element::host(if *missing.get() { 8 } else { 7 })
                }),
                Element::host(9),
            )
        };
        let mut root = mount(element);
        assert_eq!(top_level(&root), [Some(7)]);
        let Some(missing) = handle.borrow().clone() else {
            return;
        };

        // The update hides content the user is looking at, so the fallback is
        // held for the first noticeable difference instead of replacing the
        // screen at once. Nothing on screen changes and the held tree is work
        // the root still owes.
        missing.set(true);
        assert!(root.flush_slice(0, || false));
        assert_eq!(top_level(&root), [Some(7)]);
        assert!(root.has_pending_work());

        assert!(root.flush_slice(119, || false));
        assert_eq!(top_level(&root), [Some(7)]);

        // The boundary is reached, so the fallback commits beside the primary
        // tree, which stays in the host hidden.
        assert!(!root.flush_slice(120, || false));
        assert_eq!(top_level(&root), [Some(7), Some(9)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_delayed_fallback_measures_its_wait_from_the_update() {
        let shell_missing = Rc::new(Cell::new(false));
        let content_ready = Rc::new(Cell::new(true));
        let handle = Rc::new(RefCell::new(None::<State<bool>>));
        let (outer_work, outer_subscribers) = manual();
        let (inner_work, _inner_subscribers) = manual();
        let element = {
            let shell_missing = Rc::clone(&shell_missing);
            let content_ready = Rc::clone(&content_ready);
            let handle = Rc::clone(&handle);
            let outer_work = outer_work.clone();
            let inner_work = inner_work.clone();
            Element::component(move |ctx| {
                let triggered = ctx.use_state(|| false);
                *handle.borrow_mut() = Some(triggered.clone());
                if *triggered.get() && shell_missing.get() {
                    return ctx.suspend(outer_work.clone());
                }
                Element::suspense(waiting_on(&content_ready, &inner_work), Element::host(9))
            })
        };
        let mut root = mount(element);
        assert_eq!(top_level(&root), [Some(7)]);
        let Some(triggered) = handle.borrow().clone() else {
            return;
        };

        // The update suspends at the shell, so the previous screen stays and
        // the lane parks.
        content_ready.set(false);
        shell_missing.set(true);
        triggered.set(true);
        root.flush_slice(0, || false);
        assert_eq!(top_level(&root), [Some(7)]);
        assert_eq!(root.suspended_lanes, Lanes::DEFAULT);

        // The work resolves 500 ms into the wait. The retry reaches the
        // boundary this time, and the fallback it would show measures the
        // wait from the update at 0: the next boundary above 500 is 1080, not
        // the first one.
        shell_missing.set(false);
        resolve(&outer_subscribers);
        assert!(root.flush_slice(500, || false));
        assert_eq!(top_level(&root), [Some(7)]);
        assert!(root.has_pending_work());

        assert!(root.flush_slice(1_079, || false));
        assert_eq!(top_level(&root), [Some(7)]);

        assert!(!root.flush_slice(1_080, || false));
        assert_eq!(top_level(&root), [Some(7), Some(9)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn each_boundary_retry_claims_a_fresh_lane() {
        let first_ready = Rc::new(Cell::new(false));
        let second_ready = Rc::new(Cell::new(false));
        let (first_work, first_subscribers) = manual();
        let (second_work, second_subscribers) = manual();
        let mut root = mount(Element::fragment([
            Element::suspense(waiting_on(&first_ready, &first_work), Element::host(9)),
            Element::suspense(waiting_on(&second_ready, &second_work), Element::host(8)),
        ]));
        assert_eq!(top_level(&root), [Some(9), Some(8)]);

        // Each retry takes the next lane of the group, so a retry that parks
        // its lane cannot hold back one that starts later.
        assert_eq!(root.next_retry_lane, Lanes::RETRY_FIRST);
        first_ready.set(true);
        resolve(&first_subscribers);
        root.flush();
        assert_eq!(top_level(&root), [Some(7), Some(8)]);
        assert_eq!(root.next_retry_lane, Lanes::RETRY_FIRST.next_retry());

        second_ready.set(true);
        resolve(&second_subscribers);
        root.flush();
        assert_eq!(top_level(&root), [Some(7), Some(7)]);
        assert_eq!(root.next_retry_lane, Lanes::RETRY_FIRST.next_retry().next_retry());
        assert!(!root.has_pending_work());
    }

    #[test]
    fn idle_work_waits_behind_a_suspended_lane() {
        let handle = Rc::new(RefCell::new(None));
        let rendered = Rc::new(Cell::new(0));
        let mut root = mount(sometimes(&handle, &rendered));
        let Some(missing) = handle.borrow().clone() else {
            return;
        };
        missing.set(true);
        root.flush();
        assert_eq!(root.suspended_lanes, Lanes::DEFAULT);

        // The idle update is the only work that is not parked, and idle work
        // does not overtake a suspended lane: a render would commit the tree
        // around the parked lane as if its work had finished.
        missing.set_with_lane(Lanes::IDLE, false);
        root.flush();
        assert_eq!(top_level(&root), [Some(7)]);
        assert_eq!(root.pending_lanes, Lanes::DEFAULT | Lanes::IDLE);
        assert!(!root.has_pending_work());
        assert_eq!(rendered.get(), 2);

        // The default update unblocks the parked lane; the idle update follows
        // in the same flush, once no non-idle lane is left.
        missing.set(false);
        root.flush();
        assert_eq!(rendered.get(), 4);
        assert!(root.pending_lanes.is_empty());
        assert!(!root.has_pending_work());
    }

    /// The nodes the container shows, in order.
    fn mounted(root: &Root<TreeHost>) -> Vec<NodeId> {
        root.host()
            .children(root.host().container())
            .to_vec()
    }

    #[test]
    fn a_reference_receives_the_node_when_it_mounts() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let reference: Reference<TreeHost> = {
            let calls = Rc::clone(&calls);
            Rc::new(move |node| {
                calls.borrow_mut().push(node);
                None
            })
        };
        let root = mount(Element::host(1).referenced(Rc::clone(&reference)));

        assert_eq!(*calls.borrow(), [Some(mounted(&root)[0])]);
    }

    #[test]
    fn a_reference_detaches_when_its_node_unmounts() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let reference: Reference<TreeHost> = {
            let calls = Rc::clone(&calls);
            Rc::new(move |node| {
                calls.borrow_mut().push(node);
                None
            })
        };
        let mut root = mount(Element::host(1).referenced(Rc::clone(&reference)));
        let node = mounted(&root)[0];

        root.render(Element::empty());
        root.flush();
        assert!(top_level(&root).is_empty());
        assert_eq!(*calls.borrow(), [Some(node), None]);
    }

    #[test]
    fn a_changed_reference_detaches_the_old_callback_before_attaching_the_new() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let make = |tag: &'static str| -> Reference<TreeHost> {
            let calls = Rc::clone(&calls);
            Rc::new(move |node| {
                calls.borrow_mut().push((tag, node));
                None
            })
        };
        let first = make("first");
        let second = make("second");
        let mut root = mount(Element::host(1).referenced(Rc::clone(&first)));
        let node = mounted(&root)[0];

        root.render(Element::host(1).referenced(Rc::clone(&second)));
        root.flush();
        assert_eq!(
            *calls.borrow(),
            [("first", Some(node)), ("first", None), ("second", Some(node))]
        );
    }

    #[test]
    fn a_reference_that_does_not_change_is_not_re_attached() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let reference: Reference<TreeHost> = {
            let calls = Rc::clone(&calls);
            Rc::new(move |node| {
                calls.borrow_mut().push(node);
                None
            })
        };
        let element = {
            let handle = Rc::clone(&handle);
            let log = Rc::clone(&log);
            let reference = Rc::clone(&reference);
            Element::component(move |ctx| {
                let value = ctx.use_state(|| 1_u32);
                *handle.borrow_mut() = Some(value.clone());
                log.borrow_mut().push(*value.get());
                // The node keeps its kind across the update, so the fiber is
                // reused and only the ref comparison can ask for a re-attach.
                Element::host(7).referenced(Rc::clone(&reference))
            })
        };
        let mut root = mount(element);
        assert_eq!(calls.borrow().len(), 1);

        // The re-render rebuilds the host element but keeps the callback, so
        // the node is neither detached nor attached again.
        let Some(state) = handle.borrow().clone() else {
            return;
        };
        state.set(2);
        root.flush();
        assert_eq!(*log.borrow(), [1, 2]);
        assert_eq!(top_level(&root), [Some(7)]);
        assert_eq!(calls.borrow().len(), 1);
    }

    #[test]
    fn a_cleanup_returned_by_a_reference_runs_instead_of_the_none_call() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let cleaned = Rc::new(Cell::new(0_u32));
        let reference: Reference<TreeHost> = {
            let calls = Rc::clone(&calls);
            let cleaned = Rc::clone(&cleaned);
            Rc::new(move |node| match node {
                Some(_) => {
                    let cleaned = Rc::clone(&cleaned);
                    Some(Box::new(move || cleaned.set(cleaned.get() + 1)) as Cleanup)
                }
                None => {
                    calls.borrow_mut().push("detached");
                    None
                }
            })
        };
        let mut root = mount(Element::host(1).referenced(Rc::clone(&reference)));

        root.render(Element::empty());
        root.flush();
        assert_eq!(cleaned.get(), 1);
        assert!(calls.borrow().is_empty());
    }

    #[test]
    fn a_state_update_reaches_the_host_on_the_next_flush() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(counter(&handle, &log));
        assert_eq!(top_level(&root), [Some(1)]);

        let Some(state) = handle.borrow().clone() else {
            return;
        };
        state.set(2);
        assert!(root.has_pending_work());
        root.flush();
        assert_eq!(top_level(&root), [Some(2)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_render_phase_update_is_committed_by_the_same_flush() {
        let log = Rc::new(RefCell::new(Vec::<u32>::new()));
        let element = {
            let log = Rc::clone(&log);
            Element::component(move |ctx| {
                let value = ctx.use_state(|| 1_u32);
                let current = *value.get();
                log.borrow_mut().push(current);
                if current < 2 {
                    value.set(2);
                }
                Element::host(current)
            })
        };
        let root = mount(element);

        // The component ran twice before its element reached the host, so the
        // host saw one creation and never an intermediate tree.
        assert_eq!(*log.borrow(), [1, 2]);
        assert_eq!(top_level(&root), [Some(2)]);
        assert_eq!(root.host().nodes[1].updates, 0);
        assert!(!root.has_pending_work());
    }

    /// A component that renders `child` and, once `dispatch` is set, writes the
    /// state of another component while it renders.
    fn parent_of(
        child: Element<TreeHost>,
        dispatch: &Rc<Cell<bool>>,
        into: &Rc<RefCell<Option<State<u32>>>>,
    ) -> Element<TreeHost> {
        let dispatch = Rc::clone(dispatch);
        let into = Rc::clone(into);
        Element::component(move |_| {
            if dispatch.replace(false)
                && let Some(state) = into.borrow().clone()
            {
                state.set(5);
            }
            child.clone()
        })
    }

    #[test]
    fn an_update_to_another_component_during_a_render_is_scheduled() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let dispatch = Rc::new(Cell::new(false));
        let child = counter(&handle, &log);
        let mut root = mount(parent_of(child.clone(), &dispatch, &handle));
        assert_eq!(*log.borrow(), [1]);

        // The write names the child, not the component being rendered, so the
        // flush turns it into a render of the child instead of an in-place
        // re-render of the parent.
        dispatch.set(true);
        root.render(parent_of(child, &dispatch, &handle));
        root.flush();
        assert_eq!(*log.borrow(), [1, 5]);
        assert_eq!(top_level(&root), [Some(5)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_sibling_with_no_new_element_does_not_render_again() {
        let first = Rc::new(RefCell::new(None::<State<u32>>));
        let first_log = Rc::new(RefCell::new(Vec::new()));
        let second = Rc::new(RefCell::new(None::<State<u32>>));
        let second_log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(Element::fragment([
            counter(&first, &first_log),
            counter(&second, &second_log),
        ]));
        assert_eq!(*first_log.borrow(), [1]);
        assert_eq!(*second_log.borrow(), [1]);

        let Some(state) = first.borrow().clone() else {
            return;
        };
        state.set(2);
        root.flush();

        assert_eq!(*first_log.borrow(), [1, 2]);
        // The sibling kept the element the root still holds and no lane of it
        // is pending, so the render never enters it.
        assert_eq!(*second_log.borrow(), [1]);
        assert_eq!(top_level(&root), [Some(2), Some(1)]);
    }

    #[test]
    fn a_new_element_renders_the_component_again() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(counter(&handle, &log));
        assert_eq!(*log.borrow(), [1]);

        // The same component type with a different element: the description
        // changed, so the component renders even though no lane is pending.
        root.render(counter(&handle, &log));
        root.flush();

        assert_eq!(*log.borrow(), [1, 1]);
        assert_eq!(top_level(&root), [Some(1)]);
    }

    #[test]
    fn a_skipped_transition_update_is_rebased_into_the_next_lane() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(counter(&handle, &log));
        let Some(state) = handle.borrow().clone() else {
            return;
        };

        // The doubling is queued first, but the default lane is more urgent:
        // its render skips the transition update and replays it in source
        // order once the transition lane gets its turn.
        state.update_with_lane(Lanes::TRANSITION.highest(), |value| *value * 2);
        state.update(|value| *value + 10);
        root.flush();

        assert_eq!(*log.borrow(), [1, 11, 12]);
        assert_eq!(top_level(&root), [Some(12)]);
    }

    #[test]
    fn idle_work_waits_for_the_default_lane() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(counter(&handle, &log));
        let Some(state) = handle.borrow().clone() else {
            return;
        };

        // The idle value is queued first; the default render still runs first,
        // and the idle lane replays both updates in source order afterwards.
        state.set_with_lane(Lanes::IDLE, 2);
        state.update(|value| *value + 10);
        root.flush();

        assert_eq!(*log.borrow(), [1, 11, 12]);
        assert_eq!(top_level(&root), [Some(12)]);
    }

    #[test]
    fn an_update_from_an_unmounted_component_is_dropped() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(counter(&handle, &log));
        let Some(state) = handle.borrow().clone() else {
            return;
        };

        root.render(Element::empty());
        root.flush();
        assert!(top_level(&root).is_empty());

        state.set(9);
        assert!(!root.has_pending_work());
        root.flush();
        assert!(top_level(&root).is_empty());
    }

    #[test]
    fn a_layout_effect_update_is_rendered_in_the_same_flush() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let once = Rc::new(Cell::new(false));
        let element = {
            let handle = Rc::clone(&handle);
            let once = Rc::clone(&once);
            Element::component(move |ctx| {
                let value = ctx.use_state(|| 1_u32);
                *handle.borrow_mut() = Some(value.clone());
                let dispatch = value.clone();
                let once = Rc::clone(&once);
                ctx.use_layout_effect((), move || {
                    if !once.replace(true) {
                        dispatch.set(2);
                    }
                    None
                });
                Element::host(*value.get())
            })
        };

        let root = mount(element);
        assert_eq!(top_level(&root), [Some(2)]);
    }

    #[test]
    fn a_passive_effect_of_a_transition_waits_for_the_passive_flush() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::<u32>::new()));
        let element = {
            let handle = Rc::clone(&handle);
            let log = Rc::clone(&log);
            Element::component(move |ctx| {
                let value = ctx.use_state(|| 1_u32);
                *handle.borrow_mut() = Some(value.clone());
                let rendered = *value.get();
                let log = Rc::clone(&log);
                ctx.use_effect(rendered, move || {
                    log.borrow_mut().push(rendered);
                    None
                });
                Element::host(rendered)
            })
        };

        let mut root = mount(element);
        // The mount is a sync-update commit, so its passive effect already ran.
        assert_eq!(*log.borrow(), [1]);
        log.borrow_mut().clear();

        let Some(state) = handle.borrow().clone() else {
            return;
        };
        state.set_with_lane(Lanes::TRANSITION.highest(), 2);
        root.flush();
        assert_eq!(top_level(&root), [Some(2)]);
        // A transition commit is not a sync-update commit: the effect waits
        // for the host to present the frame.
        assert!(log.borrow().is_empty());

        assert!(root.flush_passive());
        assert_eq!(*log.borrow(), [2]);
        assert!(!root.flush_passive());
    }

    #[test]
    fn the_passive_effects_of_the_previous_commit_run_before_the_next_one() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::<(u32, &'static str)>::new()));
        let element = {
            let handle = Rc::clone(&handle);
            let log = Rc::clone(&log);
            Element::component(move |ctx| {
                let value = ctx.use_state(|| 1_u32);
                *handle.borrow_mut() = Some(value.clone());
                let rendered = *value.get();
                log.borrow_mut().push((rendered, "render"));
                let log = Rc::clone(&log);
                ctx.use_effect(rendered, move || {
                    log.borrow_mut().push((rendered, "passive"));
                    None
                });
                Element::host(rendered)
            })
        };

        let mut root = mount(element);
        assert_eq!(*log.borrow(), [(1, "render"), (1, "passive")]);
        log.borrow_mut().clear();

        let Some(state) = handle.borrow().clone() else {
            return;
        };
        // The transition commit leaves its passive effect pending.
        state.set_with_lane(Lanes::TRANSITION.highest(), 2);
        root.flush();
        assert_eq!(*log.borrow(), [(2, "render")]);
        log.borrow_mut().clear();

        // The next commit renders first, then runs the pending passive effect
        // of the tree it is about to replace, then commits and runs its own.
        state.set(3);
        root.flush();
        assert_eq!(*log.borrow(), [(3, "render"), (2, "passive"), (3, "passive")]);
    }

    #[test]
    fn a_time_sliced_render_reaches_the_host_only_when_it_completes() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(counter(&handle, &log));
        let Some(state) = handle.borrow().clone() else {
            return;
        };
        state.set_with_lane(Lanes::TRANSITION.highest(), 2);

        // The slice stops after the root, before the component runs: the
        // work-in-progress tree exists but the host still shows the old one.
        let mut checks = 0;
        assert!(root.flush_slice(0, || {
            checks += 1;
            checks > 1
        }));
        assert_eq!(*log.borrow(), [1]);
        assert_eq!(top_level(&root), [Some(1)]);
        assert!(root.has_pending_work());

        root.flush();
        assert_eq!(*log.borrow(), [1, 2]);
        assert_eq!(top_level(&root), [Some(2)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_blocking_render_ignores_the_yield_callback() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = scheduled(counter(&handle, &log));

        // Default work is never time sliced, so the callback is not even asked.
        let mut asked = false;
        assert!(!root.flush_slice(0, || {
            asked = true;
            true
        }));
        assert!(!asked);
        assert_eq!(top_level(&root), [Some(1)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn more_urgent_work_restarts_a_render_in_flight() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(counter(&handle, &log));
        let Some(state) = handle.borrow().clone() else {
            return;
        };

        // The transition render reaches the component and stops right after
        // it, so the value it read never reaches the host.
        state.update_with_lane(Lanes::TRANSITION.highest(), |value| *value * 10);
        let mut checks = 0;
        assert!(root.flush_slice(0, || {
            checks += 1;
            checks > 2
        }));
        assert_eq!(*log.borrow(), [1, 10]);
        assert_eq!(top_level(&root), [Some(1)]);

        // A sync update outranks the transition, so the render is thrown away
        // and starts again from the committed tree. The abandoned update is
        // not lost: the transition replays it on top of the sync one.
        state.update_with_lane(Lanes::SYNC, |value| *value + 5);
        root.flush();
        assert_eq!(*log.borrow(), [1, 10, 6, 15]);
        assert_eq!(top_level(&root), [Some(15)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn work_that_arrives_during_a_render_stays_pending() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(counter(&handle, &log));
        let Some(state) = handle.borrow().clone() else {
            return;
        };

        state.set_with_lane(Lanes::TRANSITION.highest(), 2);
        let mut checks = 0;
        assert!(root.flush_slice(0, || {
            checks += 1;
            checks > 2
        }));
        assert_eq!(*log.borrow(), [1, 2]);

        // Default work does not interrupt a transition, so the render in flight
        // finishes first. The render is already past the fiber the update names,
        // so the completion has to carry the lane by itself instead of trusting
        // the finished tree to hold it.
        state.set(5);
        root.flush();
        assert_eq!(*log.borrow(), [1, 2, 5]);
        assert_eq!(top_level(&root), [Some(5)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn a_starved_transition_finishes_in_one_slice() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(counter(&handle, &log));
        let Some(state) = handle.borrow().clone() else {
            return;
        };

        state.set_with_lane(Lanes::TRANSITION.highest(), 2);
        // The lane has just started waiting, so the render is still sliceable.
        let mut asked = false;
        assert!(root.flush_slice(0, || {
            asked = true;
            true
        }));
        assert!(asked);
        assert_eq!(top_level(&root), [Some(1)]);

        // Five seconds later the transition has waited its full delay. The
        // render completes in this slice even though the host asks to stop.
        asked = false;
        assert!(!root.flush_slice(5_000, || {
            asked = true;
            true
        }));
        assert!(!asked);
        assert_eq!(*log.borrow(), [1, 2]);
        assert_eq!(top_level(&root), [Some(2)]);
        assert!(!root.has_pending_work());
    }

    #[test]
    fn an_expiration_time_is_cleared_when_the_lane_is_rendered() {
        let handle = Rc::new(RefCell::new(None::<State<u32>>));
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut root = mount(counter(&handle, &log));
        let Some(state) = handle.borrow().clone() else {
            return;
        };

        // The first update starts the clock and the render yields, so the lane
        // expires while the render is still in flight.
        state.set_with_lane(Lanes::TRANSITION.highest(), 2);
        assert!(root.flush_slice(1_000, || true));
        assert!(!root.flush_slice(6_000, || true));
        assert_eq!(top_level(&root), [Some(2)]);

        // The lane is no longer pending, so the expiration time is gone: the
        // update queued on the same lane gets a fresh delay instead of being
        // expired the moment the root sees it.
        state.set_with_lane(Lanes::TRANSITION.highest(), 3);
        let mut asked = false;
        assert!(root.flush_slice(6_500, || {
            asked = true;
            true
        }));
        assert!(asked);
        assert_eq!(top_level(&root), [Some(2)]);

        root.flush();
        assert_eq!(top_level(&root), [Some(3)]);
        assert!(!root.has_pending_work());
    }
}
