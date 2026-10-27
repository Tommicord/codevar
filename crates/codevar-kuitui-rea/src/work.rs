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
//! can be bubbled to the parent.
//!
//! ```text
//! begin(root)
//!   begin(component) ── render, reconcile
//!     begin(host)    ── reconcile children (none)
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
//! # Flags
//!
//! Completion is also where flags move upwards. A child's flags and lanes are
//! merged into its parent's subtree bookkeeping, so the commit phase can skip
//! a whole subtree with one test and the scheduler can find pending work
//! without touching every fiber.

#![allow(dead_code)]

use alloc::rc::Rc;
use core::mem;

use crate::element::Element;
use crate::fiber::{Arena, FiberId, FiberTag, Flags, Payload};
use crate::hooks::{EffectTag, MailboxHandle, RenderCtx};
use crate::host::Host;
use crate::lane::Lanes;
use crate::reconcile::reconcile_children;

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

    /// Renders the whole tree below `root`.
    pub(crate) fn run(&mut self, root: FiberId) {
        let mut next = self.begin_work(root);
        while let Some(fiber) = next {
            next = match self.begin_work(fiber) {
                Some(child) => Some(child),
                None => self.complete_work(fiber, root),
            };
        }
    }

    /// Renders one fiber and returns the child to work on next, if any.
    fn begin_work(&mut self, fiber: FiberId) -> Option<FiberId> {
        let tag = self.arena.get(fiber)?.tag;
        match tag {
            FiberTag::Component => self.render_component(fiber),
            FiberTag::Fragment => self.render_fragment(fiber),
            // The root's children are reconciled by the caller before the
            // pass starts; a host node's content is the host's business.
            FiberTag::Root | FiberTag::Host => self.arena.get(fiber)?.first_child,
        }
    }

    /// Runs a component and diffs the element it returned.
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
        let element = {
            let mut ctx = RenderCtx::new(
                fiber,
                &mut hooks,
                &mut effects,
                self.render_lanes,
                Rc::clone(&self.mailbox),
                &mut *self.ids,
            );
            let element = render(&mut ctx);
            ctx.finish();
            element
        };
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

    /// Diffs `children` against the committed children of `fiber`.
    fn reconcile(&mut self, fiber: FiberId, children: &[Element<H>]) {
        let current_first = self
            .arena
            .get(fiber)
            .and_then(|fiber| fiber.alternate)
            .and_then(|alternate| self.arena.get(alternate))
            .and_then(|fiber| fiber.first_child);
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
            let parent = self
                .arena
                .get(current)
                .and_then(|fiber| fiber.parent);
            if let Some(parent) = parent
                && let Some(fiber) = self.arena.get(current)
            {
                let flags = fiber.flags | fiber.subtree_flags;
                let lanes = fiber.lanes | fiber.child_lanes;
                if let Some(parent) = self.arena.get_mut(parent) {
                    parent.merge_child(flags, lanes);
                }
            }
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
            current = parent?;
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
        let node = work.state_node;
        let mut flags = Flags::NONE;
        let mut state_node = node;
        if tag == FiberTag::Host
            && let Payload::Host(element) = &payload
        {
            match node {
                Some(node) => {
                    let changed = match &memoized {
                        Payload::Host(old) => !Rc::ptr_eq(old, element),
                        _ => true,
                    };
                    if changed {
                        self.host.update(node, element);
                        flags.insert(Flags::UPDATE);
                    }
                }
                None => {
                    state_node = Some(self.host.create(element));
                    flags.insert(Flags::PLACEMENT);
                }
            }
        }
        if let Some(work) = self.arena.get_mut(fiber) {
            work.state_node = state_node;
            work.memoized = payload;
            work.flags.insert(flags);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let handle = Rc::clone(&props);
        let element = Element::component(move |_| Element::host((0, handle.get())));
        harness.render(element.clone());
        props.set(2);
        harness.render(element);
        assert_eq!(harness.host.nodes.len(), 1);
        assert_eq!(harness.host.nodes[0].element, (0, 2));
        assert_eq!(harness.host.nodes[0].updates, 1);
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
        harness.render(element);
        assert_eq!(harness.host.nodes[0].element, (0, 2));
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
        let handle = Rc::clone(&count);
        let element = Element::component(move |_| {
            let mut children = Vec::new();
            for index in 0..handle.get() {
                children.push(Element::host((0, index + 1)));
            }
            Element::fragment(children)
        });
        harness.render(element.clone());
        count.set(2);
        harness.render(element);
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
}
