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

//! The render pass.
//!
//! [`WorkLoop::run`] walks the work-in-progress tree from its root and gives
//! every fiber its turn. A fiber is *begun* before its children — that is
//! where a component runs and where the element tree is diffed into fibers —
//! and *completed* after them, when its children have a host node and flags
//! can be bubbled to the parent. One begun fiber and the chain of ancestors
//! that completes with it make one unit of work, which is also the smallest
//! step a render can be stopped after (see [`WorkLoop::step`]).
//!
//! ```text
//! begin(root)
//!   begin(component) ── render, reconcile
//!     begin(host)    ── reconcile the children it arranges
//!     complete(host) ── create or update the node
//!   complete(component)
//! complete(root)
//! ```
//!
//! The loop keeps one cursor instead of a call stack: when a fiber has no
//! child, it completes and moves to its next sibling, or climbs to the parent
//! when it has none. Because the tree is stored in an arena, the climb is a
//! field read rather than a stack pop.
//!
//! # Host nodes
//!
//! A host node is created in the complete pass, which is the first moment the
//! fiber knows it survives the diff. Reused host fibers keep their node; the
//! elements are compared and the node is updated in place, with `Rc` identity
//! deciding whether there is anything to update at all.
//!
//! # Flags and lanes
//!
//! Completion is also where flags and lanes move upwards. The union over the
//! children replaces the parent's `child_lanes` and joins its `subtree_flags`,
//! the way React's `bubbleProperties` does, so the commit phase can skip a
//! whole subtree with one flag test and the scheduler can find pending work
//! without touching every fiber. The lanes a component could not apply are
//! left on its fiber by [`RenderCtx`]; bubbling is what carries them to the
//! root, where `finishedWork.lanes | finishedWork.child_lanes` is the work the
//! next render has to cover.

#![allow(dead_code)]

use alloc::rc::Rc;
use core::mem;

use crate::element::Element;
use crate::fiber::{Arena, FiberId, FiberTag, Flags, Payload};
use crate::hooks::{EffectTag, MailboxHandle, RenderCtx};
use crate::host::Host;
use crate::lane::Lanes;
use crate::reconcile::{committed_first_child, reconcile_children};

/// The state the render pass needs while it walks the tree.
pub(crate) struct WorkLoop<'a, H: Host> {
    /// The host the nodes are created in.
    host: &'a mut H,
    /// The arena that owns the fibers.
    arena: &'a mut Arena<H>,
    /// The mailbox a dispatch signals.
    mailbox: MailboxHandle,
    /// The counter of hook ids.
    ids: &'a mut u64,
    /// The lanes this pass renders.
    render_lanes: Lanes,
}

/// The number of extra passes a component may ask for before the render stops.
///
/// React throws "Too many re-renders" at this limit. Rea has no error channel,
/// so the loop stops instead and drops the updates the last pass queued: they
/// were queued for a pass the limit did not allow.
const RE_RENDER_LIMIT: u32 = 25;

impl<'a, H: Host> WorkLoop<'a, H> {
    /// Creates a render pass.
    pub(crate) fn new(
        host: &'a mut H,
        arena: &'a mut Arena<H>,
        mailbox: MailboxHandle,
        ids: &'a mut u64,
        render_lanes: Lanes,
    ) -> Self {
        Self {
            host,
            arena,
            mailbox,
            ids,
            render_lanes,
        }
    }

    /// Renders one unit of work and returns the next fiber to begin.
    ///
    /// A unit is one fiber begun and, when it has no child, the chain of
    /// ancestors completed with it, so a component is never left half-rendered.
    /// The cursor is the only state the pass keeps: a caller that stops between
    /// units resumes by passing the returned fiber, which is what time slicing
    /// does.
    pub(crate) fn step(&mut self, fiber: FiberId, root: FiberId) -> Option<FiberId> {
        match self.begin_work(fiber) {
            Some(child) => Some(child),
            None => self.complete_work(fiber, root),
        }
    }

    /// Renders the whole tree below `root`.
    ///
    /// The root is begun like every other fiber, so it is also completed when
    /// it has no child left — which is what replaces the lanes and flags the
    /// work-in-progress copy inherited from the committed tree.
    pub(crate) fn run(&mut self, root: FiberId) {
        let mut next = Some(root);
        while let Some(fiber) = next {
            next = self.step(fiber, root);
        }
    }

    /// Renders one fiber and returns the child to work on next, if any.
    fn begin_work(&mut self, fiber: FiberId) -> Option<FiberId> {
        let node = self.arena.get(fiber)?;
        let tag = node.tag;
        if let Some(current) = node.alternate
            && !self.did_receive_update(fiber, current)
        {
            return self.bailout_on_already_finished_work(fiber);
        }
        if let Some(node) = self.arena.get_mut(fiber) {
            // The lanes copied from the committed alternate describe the work
            // the last render found. A fiber that renders replaces them with
            // what this render could not apply.
            node.lanes = Lanes::NONE;
        }
        match tag {
            FiberTag::Component => self.render_component(fiber),
            FiberTag::Fragment => self.render_fragment(fiber),
            FiberTag::Host => self.render_host(fiber),
            // The root's children are reconciled by the caller before the pass
            // starts.
            FiberTag::Root => self.arena.get(fiber)?.first_child,
        }
    }

    /// Whether the fiber has an element of its own to apply or work of its own
    /// lane to replay.
    ///
    /// This is React's `didReceiveUpdate`. The element decides first: a fiber
    /// whose element did not change still renders when a lane it owns belongs
    /// to this render, and one whose element changed renders even without a
    /// lane of its own.
    fn did_receive_update(&self, fiber: FiberId, current: FiberId) -> bool {
        let (Some(work), Some(current)) = (self.arena.get(fiber), self.arena.get(current)) else {
            return true;
        };
        !current.memoized.same(&work.pending) || current.lanes.intersects(self.render_lanes)
    }

    /// Skips a fiber whose element and lanes need no work.
    ///
    /// When no lane below the fiber belongs to this render, the subtree below
    /// it is already what the host shows, so the pass leaves it alone and the
    /// finished tree keeps sharing it with the committed one. Otherwise the
    /// pass still has to walk in, and the children are cloned first: the fiber
    /// only has a link to the committed child list, and walking that list
    /// would write to the committed tree.
    fn bailout_on_already_finished_work(&mut self, fiber: FiberId) -> Option<FiberId> {
        let node = self.arena.get(fiber)?;
        if !node.child_lanes.intersects(self.render_lanes) {
            return None;
        }
        match node.tag {
            // The caller reconciles the root's child before the pass starts,
            // so the child it would clone is already a work-in-progress fiber.
            FiberTag::Root => node.first_child,
            FiberTag::Component | FiberTag::Fragment | FiberTag::Host => self.clone_child_fibers(fiber),
        }
    }

    /// Clones the committed children of a bailed-out fiber.
    ///
    /// The clone keeps the committed payload as its pending one: a bailed-out
    /// fiber never reconciled, so its children still describe what the host
    /// shows. React's `cloneChildFibers` preserves `pendingProps` the same way.
    fn clone_child_fibers(&mut self, fiber: FiberId) -> Option<FiberId> {
        let first = self.arena.get(fiber)?.first_child?;
        let mut current = Some(first);
        let mut cloned = None;
        let mut previous: Option<FiberId> = None;
        while let Some(child) = current {
            let Some(work) = self.arena.create_work_in_progress(child) else {
                break;
            };
            if let Some(node) = self.arena.get_mut(work) {
                node.parent = Some(fiber);
            }
            match previous {
                Some(previous) => {
                    if let Some(node) = self.arena.get_mut(previous) {
                        node.sibling = Some(work);
                    }
                }
                None => {
                    if let Some(node) = self.arena.get_mut(fiber) {
                        node.first_child = Some(work);
                    }
                    cloned = Some(work);
                }
            }
            previous = Some(work);
            current = self
                .arena
                .get(child)
                .and_then(|fiber| fiber.sibling);
        }
        cloned
    }

    /// Runs a component and diffs the element it returned.
    ///
    /// A component that writes to its own state while it renders asks for
    /// another pass. React re-renders it in place, before the element reaches
    /// the reconciler, so the intermediate element is never mounted; the loop
    /// here does the same. Every pass starts from the same committed hook list
    /// and compares its effects against the committed dependencies, so the
    /// effects of the last pass are the ones the commit runs.
    fn render_component(&mut self, fiber: FiberId) -> Option<FiberId> {
        let Some(Payload::Component(render)) = self
            .arena
            .get(fiber)
            .map(|fiber| fiber.pending.clone())
        else {
            return self.arena.get(fiber)?.first_child;
        };
        let mut hooks = self
            .arena
            .get_mut(fiber)
            .map(|fiber| mem::take(&mut fiber.hooks))
            .unwrap_or_default();
        let mut effects = self
            .arena
            .get_mut(fiber)
            .map(|fiber| mem::take(&mut fiber.effects))
            .unwrap_or_default();
        hooks.begin_render();
        let alternate = self
            .arena
            .get(fiber)
            .and_then(|fiber| fiber.alternate);
        self.mailbox
            .borrow_mut()
            .begin_render(fiber, alternate, self.render_lanes);
        let mut lanes = Lanes::NONE;
        let mut element;
        let mut rerenders = 0;
        loop {
            effects.clear();
            element = {
                let mut ctx = RenderCtx::new(
                    fiber,
                    &mut hooks,
                    &mut effects,
                    self.render_lanes,
                    &mut lanes,
                    Rc::clone(&self.mailbox),
                    &mut *self.ids,
                );
                let element = render(&mut ctx);
                ctx.finish();
                element
            };
            if !self
                .mailbox
                .borrow_mut()
                .take_render_phase_update()
            {
                break;
            }
            if rerenders >= RE_RENDER_LIMIT {
                hooks.discard_render_phase_updates(self.render_lanes);
                break;
            }
            rerenders += 1;
        }
        self.mailbox.borrow_mut().end_render();
        let mut flags = Flags::NONE;
        for effect in &effects {
            match effect.tag() {
                EffectTag::Layout => flags.insert(Flags::LAYOUT),
                EffectTag::Passive => flags.insert(Flags::PASSIVE),
            }
        }
        if let Some(current) = self.arena.get_mut(fiber) {
            current.hooks = hooks;
            current.effects = effects;
            current.flags.insert(flags);
            current.lanes.insert(lanes);
        }
        self.reconcile(fiber, &[element]);
        self.arena.get(fiber)?.first_child
    }

    /// Diffs the children a fragment describes.
    fn render_fragment(&mut self, fiber: FiberId) -> Option<FiberId> {
        let Some(Payload::Fragment(children)) = self
            .arena
            .get(fiber)
            .map(|fiber| fiber.pending.clone())
        else {
            return self.arena.get(fiber)?.first_child;
        };
        self.reconcile(fiber, &children);
        self.arena.get(fiber)?.first_child
    }

    /// Diffs the children a host node arranges.
    fn render_host(&mut self, fiber: FiberId) -> Option<FiberId> {
        let Some(Payload::Host(element)) = self
            .arena
            .get(fiber)
            .map(|fiber| fiber.pending.clone())
        else {
            return self.arena.get(fiber)?.first_child;
        };
        self.reconcile(fiber, &element.children);
        self.arena.get(fiber)?.first_child
    }

    /// Diffs `children` against the committed children of `fiber`.
    fn reconcile(&mut self, fiber: FiberId, children: &[Element<H>]) {
        let current_first = committed_first_child(self.arena, fiber);
        reconcile_children(self.host, self.arena, fiber, current_first, children);
    }

    /// Completes a fiber and every ancestor that has no work left.
    ///
    /// Returns the next fiber to begin: the next sibling of the first ancestor
    /// that has one, or `None` when the root is done.
    fn complete_work(&mut self, fiber: FiberId, root: FiberId) -> Option<FiberId> {
        let mut current = fiber;
        loop {
            self.complete(current);
            if let Some(sibling) = self
                .arena
                .get(current)
                .and_then(|fiber| fiber.sibling)
            {
                return Some(sibling);
            }
            if current == root {
                return None;
            }
            current = self
                .arena
                .get(current)
                .and_then(|fiber| fiber.parent)?;
        }
    }

    /// Bubbles the flags and lanes of the children into `fiber`.
    ///
    /// `child_lanes` is overwritten rather than merged: every child of the
    /// fiber is complete when this runs, so the union over the children is the
    /// complete set of lanes below the node for this render. Keeping the lanes
    /// copied from the committed alternate would carry a lane that was already
    /// rendered into every later render.
    fn bubble(&mut self, fiber: FiberId) {
        let Some(node) = self.arena.get(fiber) else {
            return;
        };
        let mut child = node.first_child;
        let mut flags = Flags::NONE;
        let mut lanes = Lanes::NONE;
        while let Some(current) = child {
            let Some(node) = self.arena.get_mut(current) else {
                break;
            };
            // A child the pass did not clone is shared with the committed tree
            // and still points at its committed parent. The finished tree owns
            // it now, which is the link a later dispatch climbs to mark work.
            // React repairs the pointer the same way in `bubbleProperties`.
            node.parent = Some(fiber);
            flags.insert(node.flags);
            flags.insert(node.subtree_flags);
            lanes.insert(node.lanes.union(node.child_lanes));
            child = node.sibling;
        }
        if let Some(node) = self.arena.get_mut(fiber) {
            node.subtree_flags.insert(flags);
            node.child_lanes = lanes;
        }
    }

    /// Applies the fiber itself: creates or updates its host node and makes
    /// the pending payload the committed one.
    fn complete(&mut self, fiber: FiberId) {
        let Some(work) = self.arena.get(fiber) else {
            return;
        };
        let tag = work.tag;
        let payload = work.pending.clone();
        let memoized = work.memoized.clone();
        let state = work.state_node;
        let alternate = work.alternate;
        let mut flags = Flags::NONE;
        let mut state_node = state;
        if tag == FiberTag::Host
            && let Payload::Host(element) = &payload
        {
            // React's `markRef`: a node attaches its reference when it mounts
            // or when the callback changes identity. The committed fiber is the
            // one that holds the current ref, so a fiber without an alternate
            // has none, even when reconciliation already seeded `memoized` with
            // the pending element.
            let old = alternate
                .and_then(|current| self.arena.get(current))
                .and_then(|current| current.memoized.reference())
                .cloned();
            let ref_changed = match (old.as_ref(), element.reference.as_ref()) {
                (None, None) => false,
                (Some(old), Some(new)) => !Rc::ptr_eq(old, new),
                _ => true,
            };
            if ref_changed {
                flags.insert(Flags::REF);
            }
            match state {
                Some(node) => {
                    let changed = match &memoized {
                        Payload::Host(old) => !Rc::ptr_eq(old, element),
                        _ => true,
                    };
                    if changed {
                        self.host.update(node, &element.element);
                        flags.insert(Flags::UPDATE);
                    }
                }
                None => {
                    state_node = Some(self.host.create(&element.element));
                    flags.insert(Flags::PLACEMENT);
                }
            }
        }
        if let Some(work) = self.arena.get_mut(fiber) {
            work.state_node = state_node;
            work.memoized = payload;
            work.flags.insert(flags);
        }
        self.bubble(fiber);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Reference;
    use crate::fiber::Fiber;
    use crate::hooks::{State, mailbox};
    use alloc::vec::Vec;
    use core::cell::{Cell, RefCell};

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct TestNode(u32);

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct NodeState {
        element: (u8, u16),
        updates: u32,
    }

    #[derive(Default)]
    struct TestHost {
        nodes: Vec<NodeState>,
    }

    impl Host for TestHost {
        type Element = (u8, u16);
        type Node = TestNode;

        fn element_matches(&self, old: &(u8, u16), new: &(u8, u16)) -> bool {
            old.0 == new.0
        }

        fn create(&mut self, element: &(u8, u16)) -> TestNode {
            self.nodes.push(NodeState {
                element: *element,
                updates: 0,
            });
            TestNode(self.nodes.len() as u32)
        }

        fn update(&mut self, node: TestNode, element: &(u8, u16)) {
            if let Some(state) = self.nodes.get_mut(node.0 as usize - 1) {
                state.element = *element;
                state.updates += 1;
            }
        }

        fn insert_before(&mut self, _parent: TestNode, _child: TestNode, _anchor: Option<TestNode>) {}

        fn remove(&mut self, _parent: TestNode, _child: TestNode) {}
    }

    struct Harness {
        host: TestHost,
        arena: Arena<TestHost>,
        root: FiberId,
        mailbox: MailboxHandle,
        ids: u64,
        lanes: Lanes,
    }

    impl Harness {
        fn new() -> Self {
            let mut arena = Arena::new();
            let mut fiber = Fiber::new(FiberTag::Root);
            fiber.state_node = Some(TestNode(0));
            let root = arena.alloc(fiber);
            Self {
                host: TestHost::default(),
                arena,
                root,
                mailbox: mailbox(),
                ids: 0,
                lanes: Lanes::DEFAULT,
            }
        }

        /// Renders `element` as the root's child and makes it the current tree,
        /// which is what a commit will do.
        fn render(&mut self, element: Element<TestHost>) {
            // Rendering the root's element marks the root, the way
            // `Root::render` marks it with the default lane; without the mark
            // the root would bail out and the new element would never render.
            if let Some(root) = self.arena.get_mut(self.root) {
                root.lanes.insert(self.lanes);
            }
            let Some(work) = self.arena.create_work_in_progress(self.root) else {
                return;
            };
            let current_first = self
                .arena
                .get(self.root)
                .and_then(|fiber| fiber.first_child);
            reconcile_children(&self.host, &mut self.arena, work, current_first, &[element]);
            let mut work_loop = WorkLoop::new(
                &mut self.host,
                &mut self.arena,
                Rc::clone(&self.mailbox),
                &mut self.ids,
                self.lanes,
            );
            work_loop.run(work);
            self.root = work;
        }

        fn node(&self, fiber: FiberId) -> Option<TestNode> {
            self.arena
                .get(fiber)
                .and_then(|fiber| fiber.state_node)
        }

        fn child(&self, fiber: FiberId) -> Option<FiberId> {
            self.arena
                .get(fiber)
                .and_then(|fiber| fiber.first_child)
        }

        /// Marks `lanes` on `fiber` and its ancestors, the way the root turns
        /// a drained dispatch into work.
        fn mark(&mut self, fiber: FiberId, lanes: Lanes) {
            let mut current = Some(fiber);
            let mut own = true;
            while let Some(node) = current {
                let Some(entry) = self.arena.get_mut(node) else {
                    return;
                };
                if own {
                    entry.lanes.insert(lanes);
                } else {
                    entry.child_lanes.insert(lanes);
                }
                current = entry.parent;
                own = false;
            }
        }
    }

    /// A component that renders the value it is given, with a fresh element on
    /// every call.
    fn counted(props: &Rc<Cell<u16>>) -> Element<TestHost> {
        let props = Rc::clone(props);
        Element::component(move |_| Element::host((0, props.get())))
    }

    /// A component that renders `count` host children.
    fn listing(count: &Rc<Cell<u16>>) -> Element<TestHost> {
        let count = Rc::clone(count);
        Element::component(move |_| {
            let mut children = Vec::new();
            for index in 0..count.get() {
                children.push(leaf(index + 1));
            }
            Element::fragment(children)
        })
    }

    #[test]
    fn a_component_creates_its_host_node() {
        let mut harness = Harness::new();
        harness.render(Element::component(|_| Element::host((0, 1))));
        assert_eq!(harness.host.nodes.len(), 1);
        let Some(component) = harness.child(harness.root) else {
            return;
        };
        let Some(host) = harness.child(component) else {
            return;
        };
        assert_eq!(harness.node(host), Some(TestNode(1)));
    }

    #[test]
    fn a_host_node_is_updated_instead_of_recreated() {
        let mut harness = Harness::new();
        let props = Rc::new(Cell::new(1_u16));
        harness.render(counted(&props));
        props.set(2);
        harness.render(counted(&props));
        assert_eq!(harness.host.nodes.len(), 1);
        assert_eq!(harness.host.nodes[0].element, (0, 2));
        assert_eq!(harness.host.nodes[0].updates, 1);
    }

    /// The host fiber directly below the root.
    fn host_fiber(harness: &Harness) -> Option<FiberId> {
        harness.child(harness.root)
    }

    /// Whether the host fiber carries `flag`.
    fn flagged(harness: &Harness, flag: Flags) -> bool {
        host_fiber(harness).is_some_and(|fiber| {
            harness
                .arena
                .get(fiber)
                .is_some_and(|fiber| fiber.flags.contains(flag))
        })
    }

    #[test]
    fn a_reference_is_flagged_when_it_mounts_or_changes_identity() {
        let mut harness = Harness::new();
        let first: Reference<TestHost> = Rc::new(|_| None);
        let second: Reference<TestHost> = Rc::new(|_| None);

        // A node that mounts with a callback has to attach it.
        harness.render(Element::host((0, 1)).referenced(Rc::clone(&first)));
        assert!(flagged(&harness, Flags::REF));

        // The same callback is not a change, exactly as `markRef` compares the
        // ref identity rather than the element.
        harness.render(Element::host((0, 1)).referenced(Rc::clone(&first)));
        assert!(!flagged(&harness, Flags::REF));

        harness.render(Element::host((0, 1)).referenced(Rc::clone(&second)));
        assert!(flagged(&harness, Flags::REF));

        // Dropping the callback is a change too: the committed ref must detach.
        harness.render(Element::host((0, 1)));
        assert!(flagged(&harness, Flags::REF));
    }

    #[test]
    fn a_state_update_reaches_the_host_node() {
        let mut harness = Harness::new();
        let state = Rc::new(RefCell::new(None::<State<u16>>));
        let handle = Rc::clone(&state);
        let element = Element::component(move |ctx| {
            let value = ctx.use_state(|| 1_u16);
            *handle.borrow_mut() = Some(value.clone());
            Element::host((0, *value.get()))
        });
        harness.render(element.clone());
        assert_eq!(harness.host.nodes[0].element, (0, 1));

        let Some(value) = state.borrow().clone() else {
            return;
        };
        value.set(2);
        // The root turns the dispatch into a lane on the component; the
        // element did not change, so only the lane can make it render.
        let Some(component) = harness.child(harness.root) else {
            return;
        };
        harness.mark(component, Lanes::DEFAULT);
        harness.render(element);
        assert_eq!(harness.host.nodes[0].element, (0, 2));
    }

    /// A component that renders its value and bumps it to `2` on the value it
    /// was first rendered with.
    fn bumping(log: &Rc<RefCell<Vec<u16>>>) -> Element<TestHost> {
        let log = Rc::clone(log);
        Element::component(move |ctx| {
            let value = ctx.use_state(|| 1_u16);
            let current = *value.get();
            log.borrow_mut().push(current);
            if current < 2 {
                value.set(2);
            }
            Element::host((0, current))
        })
    }

    #[test]
    fn a_render_phase_update_renders_again_before_the_element_is_reconciled() {
        let mut harness = Harness::new();
        let log = Rc::new(RefCell::new(Vec::new()));
        harness.render(bumping(&log));

        // The component runs twice, and only the element of the second pass
        // reaches the host: the first element was neither mounted nor updated.
        assert_eq!(*log.borrow(), [1, 2]);
        assert_eq!(harness.host.nodes.len(), 1);
        assert_eq!(harness.host.nodes[0].element, (0, 2));
        assert_eq!(harness.host.nodes[0].updates, 0);

        // The update was applied by the render that queued it, so the root was
        // never asked for another pass and no lane is left behind.
        assert!(harness.mailbox.borrow().is_empty());
        let Some(component) = harness.child(harness.root) else {
            return;
        };
        assert!(
            harness
                .arena
                .get(component)
                .is_some_and(|fiber| fiber.lanes.is_empty() && fiber.child_lanes.is_empty())
        );
    }

    #[test]
    fn a_render_phase_update_commits_only_the_effects_of_the_last_pass() {
        let mut harness = Harness::new();
        let element = Element::component(|ctx| {
            let value = ctx.use_state(|| 1_u16);
            let current = *value.get();
            ctx.use_effect((current,), || None);
            if current < 2 {
                value.set(2);
            }
            Element::host((0, current))
        });
        harness.render(element);
        let Some(component) = harness.child(harness.root) else {
            return;
        };
        // The first pass recorded an effect for the value it rendered; the
        // second pass replaced it, because both compare against the committed
        // dependencies. The commit therefore runs one effect per hook.
        assert_eq!(
            harness
                .arena
                .get(component)
                .map(|fiber| fiber.effects.len()),
            Some(1)
        );
    }

    #[test]
    fn a_component_that_never_stops_re_rendering_stops_at_the_limit() {
        /// A component that bumps its state on every pass it renders.
        fn bump_forever(log: &Rc<RefCell<Vec<u16>>>, dispatch: &Rc<Cell<bool>>) -> Element<TestHost> {
            let log = Rc::clone(log);
            let dispatch = Rc::clone(dispatch);
            Element::component(move |ctx| {
                let value = ctx.use_state(|| 0_u16);
                log.borrow_mut().push(*value.get());
                if dispatch.get() {
                    value.update(|current| current + 1);
                }
                Element::host((0, 1))
            })
        }

        let mut harness = Harness::new();
        let log = Rc::new(RefCell::new(Vec::new()));
        let dispatch = Rc::new(Cell::new(true));
        harness.render(bump_forever(&log, &dispatch));

        // The first pass plus the limit of re-renders, each with the value the
        // pass before queued.
        assert_eq!(log.borrow().len() as u32, RE_RENDER_LIMIT + 1);
        assert_eq!(log.borrow().last(), Some(&(RE_RENDER_LIMIT as u16)));
        assert!(harness.mailbox.borrow().is_empty());

        // The update of the last pass had no pass left to apply it and was
        // dropped, so another render starts from the value the loop stopped on
        // instead of replaying it.
        dispatch.set(false);
        harness.render(bump_forever(&log, &dispatch));
        assert_eq!(log.borrow().last(), Some(&(RE_RENDER_LIMIT as u16)));
    }

    #[test]
    fn a_skipped_update_marks_the_fiber_and_the_root() {
        let mut harness = Harness::new();
        let state = Rc::new(RefCell::new(None::<State<u16>>));
        let handle = Rc::clone(&state);
        let element = Element::component(move |ctx| {
            let value = ctx.use_state(|| 1_u16);
            *handle.borrow_mut() = Some(value.clone());
            Element::host((0, *value.get()))
        });
        harness.render(element.clone());
        let Some(value) = state.borrow().clone() else {
            return;
        };

        // The default render cannot cover the transition update: the component
        // bails out, the host keeps the rendered value, and the lane stays on
        // the fiber and reaches the root through the bubble.
        value.set_with_lane(Lanes::TRANSITION, 7);
        let Some(component) = harness.child(harness.root) else {
            return;
        };
        harness.mark(component, Lanes::TRANSITION);
        harness.render(element.clone());
        assert_eq!(harness.host.nodes[0].element, (0, 1));
        let Some(component) = harness.child(harness.root) else {
            return;
        };
        assert!(
            harness
                .arena
                .get(component)
                .is_some_and(|fiber| fiber.lanes.contains(Lanes::TRANSITION))
        );
        assert!(
            harness
                .arena
                .get(harness.root)
                .is_some_and(|fiber| fiber.child_lanes.contains(Lanes::TRANSITION))
        );

        // The transition render applies the update and leaves no lane behind.
        harness.lanes = Lanes::TRANSITION;
        harness.render(element);
        assert_eq!(harness.host.nodes[0].element, (0, 7));
        assert!(
            harness
                .arena
                .get(harness.root)
                .is_some_and(|fiber| fiber.child_lanes.is_empty())
        );
    }

    #[test]
    fn an_effect_flags_its_fiber() {
        let mut harness = Harness::new();
        harness.render(Element::component(|ctx| {
            ctx.use_effect((), || None);
            Element::host((0, 1))
        }));
        let Some(component) = harness.child(harness.root) else {
            return;
        };
        let Some(fiber) = harness.arena.get(component) else {
            return;
        };
        assert!(fiber.flags.contains(Flags::PASSIVE));
        assert_eq!(fiber.effects.len(), 1);
    }

    #[test]
    fn a_new_child_is_flagged_and_gets_a_node() {
        let mut harness = Harness::new();
        let count = Rc::new(Cell::new(1_u16));
        harness.render(listing(&count));
        count.set(2);
        harness.render(listing(&count));
        assert_eq!(harness.host.nodes.len(), 2);

        let Some(component) = harness.child(harness.root) else {
            return;
        };
        let Some(fragment) = harness.child(component) else {
            return;
        };
        let Some(first) = harness.child(fragment) else {
            return;
        };
        let second = harness
            .arena
            .get(first)
            .and_then(|fiber| fiber.sibling);
        let Some(second) = second else {
            return;
        };
        assert!(
            harness
                .arena
                .get(second)
                .is_some_and(|fiber| fiber.flags.contains(Flags::PLACEMENT))
        );
        assert!(harness.node(second).is_some());
    }

    #[test]
    fn flags_bubble_to_the_root() {
        let mut harness = Harness::new();
        harness.render(Element::component(|_| Element::host((0, 1))));
        let Some(root) = harness.arena.get(harness.root) else {
            return;
        };
        assert!(root.subtree_flags.contains(Flags::PLACEMENT));
    }

    fn leaf(props: u16) -> Element<TestHost> {
        Element::host((0, props))
    }

    fn keyed(props: u16, key: &str) -> Element<TestHost> {
        Element::host((0, props)).keyed(key)
    }

    fn container(children: Vec<Element<TestHost>>) -> Element<TestHost> {
        Element::host((0, 0)).children(children)
    }

    #[test]
    fn a_host_element_reconciles_the_children_it_carries() {
        let mut harness = Harness::new();
        harness.render(container(Vec::from([leaf(1), leaf(2)])));
        let Some(host) = harness.child(harness.root) else {
            return;
        };
        let Some(first) = harness.child(host) else {
            return;
        };
        let Some(second) = harness
            .arena
            .get(first)
            .and_then(|fiber| fiber.sibling)
        else {
            return;
        };
        assert_eq!(harness.node(first), Some(TestNode(1)));
        assert_eq!(harness.node(second), Some(TestNode(2)));
        for child in [first, second] {
            assert_eq!(
                harness
                    .arena
                    .get(child)
                    .and_then(|fiber| fiber.parent),
                Some(host)
            );
        }
    }

    #[test]
    fn host_children_keep_their_fibers_across_a_reorder() {
        let mut harness = Harness::new();
        harness.render(container(Vec::from([keyed(1, "a"), keyed(1, "b")])));
        let Some(host) = harness.child(harness.root) else {
            return;
        };
        let Some(old_a) = harness.child(host) else {
            return;
        };
        let Some(old_b) = harness
            .arena
            .get(old_a)
            .and_then(|fiber| fiber.sibling)
        else {
            return;
        };

        harness.render(container(Vec::from([keyed(1, "b"), keyed(1, "a")])));
        let Some(new_host) = harness.child(harness.root) else {
            return;
        };
        assert_eq!(
            harness
                .arena
                .get(new_host)
                .and_then(|fiber| fiber.alternate),
            Some(host)
        );
        let Some(new_b) = harness.child(new_host) else {
            return;
        };
        let Some(new_a) = harness
            .arena
            .get(new_b)
            .and_then(|fiber| fiber.sibling)
        else {
            return;
        };
        assert_eq!(
            harness
                .arena
                .get(new_b)
                .and_then(|fiber| fiber.alternate),
            Some(old_b)
        );
        assert_eq!(
            harness
                .arena
                .get(new_a)
                .and_then(|fiber| fiber.alternate),
            Some(old_a)
        );
        assert!(
            harness
                .arena
                .get(new_a)
                .is_some_and(|fiber| fiber.flags.contains(Flags::PLACEMENT))
        );
    }

    #[test]
    fn a_host_element_that_loses_its_children_reports_them() {
        let mut harness = Harness::new();
        harness.render(container(Vec::from([leaf(1), leaf(2)])));
        let Some(host) = harness.child(harness.root) else {
            return;
        };
        let Some(old) = harness.child(host) else {
            return;
        };
        let Some(second) = harness
            .arena
            .get(old)
            .and_then(|fiber| fiber.sibling)
        else {
            return;
        };

        harness.render(container(Vec::new()));
        let Some(new_host) = harness.child(harness.root) else {
            return;
        };
        assert!(
            harness
                .arena
                .get(new_host)
                .is_some_and(|fiber| fiber.deletions == [old, second])
        );
        assert!(harness.child(new_host).is_none());
    }
}
