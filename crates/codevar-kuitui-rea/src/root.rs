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
//! # Not implemented yet
//!
//! There is no suspension.

use alloc::rc::Rc;
use core::slice;

use crate::commit::Commit;
use crate::element::Element;
use crate::fiber::{Arena, Fiber, FiberId, FiberTag, Flags};
use crate::hooks::{MailboxHandle, mailbox};
use crate::host::Host;
use crate::lane::{LANE_COUNT, Lanes};
use crate::reconcile::reconcile_children;
use crate::work::WorkLoop;

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
    /// When each pending lane expires, indexed by lane bit. `None` until the
    /// root first sees the lane pending, and again once the lane is rendered.
    expiration_times: [Option<u64>; LANE_COUNT],
    /// The lanes that have been pending past their expiration delay.
    expired_lanes: Lanes,
    /// The render in flight, if a slice stopped before the tree completed.
    render: Option<Render>,
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
            expiration_times: [None; LANE_COUNT],
            expired_lanes: Lanes::NONE,
            render: None,
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

    /// Whether a dispatch or a lane is waiting for a flush.
    ///
    /// A dispatch that only reached the mailbox counts: it becomes a pending
    /// lane when [`Root::flush`] drains it.
    #[must_use]
    pub fn has_pending_work(&self) -> bool {
        !self.pending_lanes.is_empty() || !self.mailbox.borrow().is_empty()
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

    /// Renders and commits every pending lane, most urgent first.
    ///
    /// The call returns when no lane is pending and no dispatch is left in the
    /// mailbox. Work that a layout effect or a component dispatches during the
    /// call is picked up by the loop, so a caller that sets state from an
    /// effect sees the result before `flush` returns. Passive effects follow
    /// their own rule; see the module documentation.
    pub fn flush(&mut self) {
        self.flush_loop(&mut || false);
    }

    /// Renders and commits the pending lanes, giving control back between
    /// units of work.
    ///
    /// `now` is the host's monotonic clock in milliseconds. The root uses it to
    /// mark the pending lanes that have waited past [`Lanes::expiration_delay`]
    /// as expired, and a render that covers an expired lane is never stopped
    /// for the caller's budget: starved work finishes in this slice.
    ///
    /// `should_yield` is asked before every unit of a render whose lanes are
    /// neither blocking nor expired, and a `true` answer leaves the render in
    /// flight: nothing the slice built has reached the host, and the next call
    /// resumes the render at the fiber it had not started. A render that covers
    /// a blocking lane ignores the callback and always completes.
    ///
    /// The return value says whether work is left: `false` means nothing is
    /// pending, the tree committed everything it was asked to cover.
    pub fn flush_slice(&mut self, now: u64, mut should_yield: impl FnMut() -> bool) -> bool {
        // The mailbox is drained first so the lanes this call sees are the ones
        // the host's clock starts the expiration of; React marks them from
        // `ensureRootIsScheduled`, after the update has marked the root.
        self.drain_mailbox();
        self.mark_starved_lanes(now);
        self.flush_loop(&mut should_yield)
    }

    /// Renders and commits the pending lanes until none is left or a slice
    /// stops the render.
    fn flush_loop(&mut self, should_yield: &mut dyn FnMut() -> bool) -> bool {
        loop {
            self.drain_mailbox();
            let lanes = self.next_lanes();
            if lanes.is_empty() {
                return false;
            }
            if !self.perform(lanes, should_yield) {
                return true;
            }
        }
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
        let mut expired = self.expired_lanes;
        for lane in (self.pending_lanes & !Lanes::RETRY).iter() {
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
    /// Idle work waits: only when no other lane is pending does the root look
    /// at the idle, offscreen and deferred lanes. When a render is in flight,
    /// the work that arrived since it started replaces it only when strictly
    /// more urgent — and default work never interrupts a transition. This is
    /// React's rule in `getNextLanes`, which returns the lanes of the render in
    /// flight for everything else, so a render that has begun is not thrown
    /// away for equal or lower priority work.
    fn next_lanes(&self) -> Lanes {
        let non_idle = self.pending_lanes & Lanes::NON_IDLE;
        let next = if non_idle.is_empty() {
            self.pending_lanes.render_lanes()
        } else {
            non_idle.render_lanes()
        };
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
    /// for the next call; a blocking render is never stopped.
    fn perform(&mut self, lanes: Lanes, should_yield: &mut dyn FnMut() -> bool) -> bool {
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
        {
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
                next = work_loop.step(fiber, root);
            }
        }
        if next.is_some() {
            if let Some(render) = self.render.as_mut() {
                render.next = next;
            }
            return false;
        }
        self.render = None;
        self.arena.end_render();
        self.commit(root, lanes);
        true
    }

    /// Throws away a render in flight and starts a fresh tree for `lanes`.
    ///
    /// This is React's `prepareFreshStack`. The abandoned tree never reached
    /// the host: its fibers are freed, its update queues return to the state
    /// the last commit left, and the committed root is copied again. Work that
    /// arrived during the abandoned render is already in `pending_lanes`, so
    /// clearing the interleaved set here loses nothing.
    fn prepare_fresh(&mut self, lanes: Lanes) {
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
        });
    }

    /// Commits a finished tree and records the lanes the render left pending.
    fn commit(&mut self, work: FiberId, lanes: Lanes) {
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
        // A lane that is no longer pending starts over: a later update on it
        // gets a fresh expiration time instead of inheriting the one the
        // finished work had, and it is no longer expired.
        for lane in (previously_pending & !remaining).iter() {
            if let Some(index) = lane.index() {
                self.expiration_times[index as usize] = None;
            }
        }
        self.expired_lanes &= remaining;

        let mut commit = Commit::new(&mut self.host, &mut self.arena);
        commit.mutation(work);
        self.current = work;
        let mut commit = Commit::new(&mut self.host, &mut self.arena);
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
        let mut commit = Commit::new(&mut self.host, &mut self.arena);
        commit.passive(root);
        true
    }

    /// Turns the dispatches collected since the last drain into pending lanes.
    ///
    /// A queue whose fiber unmounted is detached and no longer wakes the root;
    /// should one still be signalled, its owner is `None` and the update is
    /// dropped, because a fiber that is not in the tree has no render to
    /// schedule. A dispatch that lands while a render is in flight is also
    /// recorded as interleaved: the render may be past the fiber it names, so
    /// its own lanes cannot be relied on to carry the lane to the completion.
    fn drain_mailbox(&mut self) {
        let updates = self.mailbox.borrow_mut().drain();
        for (queue, lanes) in updates {
            let Some(owner) = queue.borrow().owner() else {
                continue;
            };
            if self.render.is_some() {
                self.interleaved.insert(lanes);
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Reference;
    use crate::hooks::{Cleanup, State};
    use alloc::rc::Rc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::{Cell, RefCell};

    /// The handle of a node in [`TreeHost`].
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct NodeId(u32);

    /// One node: its element, the children the commit attached and how many
    /// times the host updated it.
    struct NodeState {
        element: Option<u32>,
        children: Vec<NodeId>,
        updates: u32,
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
