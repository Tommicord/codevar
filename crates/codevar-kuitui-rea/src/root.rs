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
//! loop, with one deliberate difference: it has no scheduler and no time
//! slicing. Work runs in the caller's turn, one lane at a time, and
//! [`Root::flush`] returns only when nothing is pending, so the host decides
//! when a frame is worth flushing.
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
//! between passes is what lets an update dispatched by a layout effect, or by
//! a component while it renders, take part in the same flush as the work that
//! caused it.
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
//! pending. React's suspension, warm-up and interruption branches have no
//! counterpart here: the root never has a render in flight, because a render
//! and its commit share one call.
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
//! # Not implemented yet
//!
//! A component always renders; the bailout that would skip a subtree with no
//! pending lanes in it is not implemented. There is no time slicing, no
//! suspension and no starvation protection. A state update dispatched while a
//! component renders is applied by the next pass of the flush loop, so the
//! intermediate tree is committed, where React re-renders the component in
//! place before committing. The hook queues already support that replay; the
//! work pass does not drive it yet.

use alloc::rc::Rc;
use core::slice;

use crate::commit::Commit;
use crate::element::Element;
use crate::fiber::{Arena, Fiber, FiberId, FiberTag, Flags};
use crate::hooks::{MailboxHandle, mailbox};
use crate::host::Host;
use crate::lane::Lanes;
use crate::reconcile::reconcile_children;
use crate::work::WorkLoop;

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
        loop {
            self.drain_mailbox();
            let lanes = self.next_lanes();
            if lanes.is_empty() {
                return;
            }
            self.perform(lanes);
        }
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
    /// at the idle, offscreen and deferred lanes. The rest of React's
    /// `getNextLanes` has no counterpart here — there is no render in flight
    /// to interrupt and no suspension to wait for.
    fn next_lanes(&self) -> Lanes {
        let non_idle = self.pending_lanes & Lanes::NON_IDLE;
        if non_idle.is_empty() {
            self.pending_lanes.render_lanes()
        } else {
            non_idle.render_lanes()
        }
    }

    /// Renders `lanes`, commits the finished tree and records what is left.
    fn perform(&mut self, lanes: Lanes) {
        let Some(work) = self.arena.create_work_in_progress(self.current) else {
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
        let mut work_loop = WorkLoop::new(
            &mut self.host,
            &mut self.arena,
            Rc::clone(&self.mailbox),
            &mut self.ids,
            lanes,
        );
        work_loop.run(work);

        // `completeRoot`: the passive effects of the previous commit run before
        // this tree becomes current, so their destroy pass still visits the
        // fibers that created them.
        while self.run_passive() {}

        let (remaining, has_passive) = match self.arena.get(work) {
            Some(fiber) => (
                fiber.lanes.union(fiber.child_lanes),
                fiber.flags.intersects(Flags::PASSIVE) || fiber.subtree_flags.intersects(Flags::PASSIVE),
            ),
            None => (Lanes::NONE, false),
        };
        // `markRootFinished`: a lane this render did not cover stays pending.
        self.pending_lanes = remaining;

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
    /// schedule.
    fn drain_mailbox(&mut self) {
        let updates = self.mailbox.borrow_mut().drain();
        for (queue, lanes) in updates {
            let Some(owner) = queue.borrow().owner() else {
                continue;
            };
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
    use crate::hooks::State;
    use alloc::rc::Rc;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::{Cell, RefCell};

    /// The handle of a node in [`TreeHost`].
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct NodeId(u32);

    /// One node: its element and the children the commit attached.
    struct NodeState {
        element: Option<u32>,
        children: Vec<NodeId>,
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
            });
            id
        }

        fn update(&mut self, node: NodeId, element: &u32) {
            if let Some(state) = self.nodes.get_mut(node.0 as usize) {
                state.element = Some(*element);
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

    /// Mounts `element` and leaves the root with no pending work.
    fn mount(element: Element<TreeHost>) -> Root<TreeHost> {
        let host = TreeHost::new();
        let container = host.container();
        let mut root = Root::new(host, container);
        root.render(element);
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
}
