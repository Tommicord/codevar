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

//! The commit phases.
//!
//! The render pass produces a tree that describes what the host should show.
//! The commit turns that description into host calls, in an order that is
//! observable and therefore fixed:
//!
//! 1. **Mutation** — removed nodes leave the host, moved nodes are placed,
//!    and the effect cleanups of the removed subtrees run.
//! 2. **Root swap** — the caller makes the finished tree current. Nothing is
//!    mutated after this point.
//! 3. **Layout** — the hook update queues are committed and every layout
//!    effect is created. A layout effect may read the host and may queue a
//!    synchronous update. Its previous cleanup already ran in the mutation
//!    phase, so all cleanups of the frame run before the first create.
//! 4. **Passive** — the previous passive effects are destroyed, then the new
//!    ones are created, in two separate walks.
//!
//! # Order inside the mutation phase
//!
//! Mutation is a post-order walk: a child is placed before its parent is.
//! That is what makes a reorder correct. A node is only "stable" — usable as
//! an anchor — once it is either already in place or never moves; with the
//! walk running children first, the anchor search simply skips every sibling
//! that is itself waiting to be placed and appends when none is left. Placing
//! in pre-order would compute anchors from nodes that are still going to move.
//!
//! Deletions are the exception: a fiber's removed children leave the host
//! before the fiber's remaining children are touched, so an insertion can
//! reuse the space and never sees a stale sibling.
//!
//! # Order inside the effect phases
//!
//! Both phases run post-order as well, so a child's effect sees the host of
//! its parent and a parent's effect sees its children already mounted. A
//! fiber's records are shared with the previous render through their destroy
//! cell, which is what lets the destroy pass of this frame run the cleanup
//! that the previous frame installed.
//!
//! # Consuming the flags
//!
//! Each phase clears the flags it applied ([`Flags::MUTATION_MASK`],
//! [`Flags::LAYOUT_MASK`], [`Flags::PASSIVE_MASK`]). The finished tree can
//! share fibers with the committed tree — a subtree with no work is not cloned
//! — and the commit walks it through those shared fibers, so a flag that
//! outlived its phase would make a later commit repeat work the host already
//! saw. Clearing is also what makes "does this tree still have passive work?"
//! answer no once the passive phase ran.

#![allow(dead_code)]

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::mem;

use crate::fiber::{Arena, FiberId, FiberTag, Flags};
use crate::hooks::EffectTag;
use crate::host::Host;

/// The passes that apply a finished tree to the host.
pub(crate) struct Commit<'a, H: Host> {
    /// The host the nodes live in.
    host: &'a mut H,
    /// The arena that owns the fibers.
    arena: &'a mut Arena<H>,
}

impl<'a, H: Host> Commit<'a, H> {
    /// Creates a commit pass.
    pub(crate) fn new(host: &'a mut H, arena: &'a mut Arena<H>) -> Self {
        Self { host, arena }
    }

    /// Applies the mutations of the finished tree: removals, then placements.
    ///
    /// The layout cleanups of the fibers that are updated also belong here, so
    /// every cleanup runs before any layout effect is created: a cleanup that
    /// reads the host sees the tree as it was when the effect ran, and one
    /// component's cleanup cannot override another component's setup.
    pub(crate) fn mutation(&mut self, root: FiberId) {
        let parent = self
            .arena
            .get(root)
            .and_then(|fiber| fiber.state_node);
        self.mutate(root, parent);
    }

    /// Commits the hook queues and creates the layout effects.
    ///
    /// The order is React's: a fiber's children are visited first, then its
    /// reference is attached, then its layout effects run. A child's ref
    /// therefore observes the node after it was placed, and a parent's layout
    /// effect observes the refs of the subtree below it.
    pub(crate) fn layout(&mut self, root: FiberId) {
        let mut order = Vec::new();
        self.collect(root, &mut order);
        for fiber in order {
            self.attach_ref(fiber);
            if let Some(node) = self.arena.get_mut(fiber) {
                if node.tag == FiberTag::Component {
                    node.hooks.commit();
                    // The swap made this fiber the current one, so a dispatch
                    // that outlives the render must name it from now on.
                    node.hooks.reset_owners(fiber);
                }
                node.flags.remove(Flags::LAYOUT_MASK);
            }
            self.create_effects(fiber, EffectTag::Layout);
        }
    }

    /// Hands the node of a finished host fiber to its reference callback.
    ///
    /// The callback may return a cleanup, which stays on the fiber — shared
    /// with its alternate — until the node is detached. A fiber without the
    /// [`Flags::REF`] flag keeps the reference it already attached: React only
    /// attaches when `markRef` saw a new or different ref.
    fn attach_ref(&mut self, fiber: FiberId) {
        let Some(node) = self.arena.get(fiber) else {
            return;
        };
        if !node.flags.contains(Flags::REF) {
            return;
        }
        let Some(reference) = node.memoized.reference().cloned() else {
            return;
        };
        let Some(state_node) = node.state_node else {
            return;
        };
        let cleanup = reference(Some(state_node));
        if let Some(node) = self.arena.get_mut(fiber) {
            node.ref_cleanup = cleanup.map(|cleanup| Rc::new(RefCell::new(Some(cleanup))));
        }
    }

    /// Destroys the previous passive effects, then creates the new ones.
    ///
    /// The two passes are separate for the same reason the layout cleanup runs
    /// in the mutation phase: every destroy of the frame runs before the first
    /// create of the frame.
    pub(crate) fn passive(&mut self, root: FiberId) {
        let mut order = Vec::new();
        self.collect(root, &mut order);
        for fiber in &order {
            self.destroy_effects(*fiber, EffectTag::Passive);
        }
        for fiber in order {
            self.create_effects(fiber, EffectTag::Passive);
            if let Some(node) = self.arena.get_mut(fiber) {
                node.flags.remove(Flags::PASSIVE_MASK);
            }
        }
    }

    /// Mutates one fiber and its subtree.
    ///
    /// `host_parent` is the node of the nearest host ancestor; a component or
    /// a fragment passes it through unchanged.
    fn mutate(&mut self, fiber: FiberId, host_parent: Option<H::Node>) {
        let Some(node) = self.arena.get(fiber) else {
            return;
        };
        let flags = node.flags;
        let own = node.state_node;
        let alternate = node.alternate;
        let children_parent = own.or(host_parent);

        if flags.contains(Flags::CHILD_DELETION) {
            let deletions = match self.arena.get_mut(fiber) {
                Some(node) => mem::take(&mut node.deletions),
                None => Vec::new(),
            };
            for deleted in deletions {
                self.unmount(deleted, children_parent);
            }
        }

        let mut child = self
            .arena
            .get(fiber)
            .and_then(|fiber| fiber.first_child);
        while let Some(current) = child {
            child = self
                .arena
                .get(current)
                .and_then(|fiber| fiber.sibling);
            self.mutate(current, children_parent);
        }

        if flags.contains(Flags::PLACEMENT)
            && let (Some(parent), Some(child)) = (host_parent, own)
        {
            let anchor = self.anchor(fiber);
            self.host.insert_before(parent, child, anchor);
        }

        // The committed ref detaches after the subtree moved and before the
        // finished ref attaches in the layout phase, which is where React
        // detaches it in `commitMutationEffectsOnFiber`.
        if flags.contains(Flags::REF)
            && let Some(current) = alternate
        {
            self.detach_ref(current);
        }

        self.destroy_effects(fiber, EffectTag::Layout);
        if let Some(node) = self.arena.get_mut(fiber) {
            node.flags.remove(Flags::MUTATION_MASK);
        }
    }

    /// The node a placement inserts before, or `None` to append.
    ///
    /// The search walks forward in tree order and skips every fiber that is
    /// itself waiting to be placed, together with its subtree. A node that is
    /// about to move cannot hold its final position, so it cannot anchor
    /// anything; only the first node that stays can.
    fn anchor(&self, fiber: FiberId) -> Option<H::Node> {
        let mut node = fiber;
        'siblings: loop {
            while self.arena.get(node)?.sibling.is_none() {
                let parent = self.arena.get(node)?.parent?;
                if self.is_host_parent(parent) {
                    return None;
                }
                node = parent;
            }
            node = self.arena.get(node)?.sibling?;
            loop {
                let current = self.arena.get(node)?;
                if current.flags.contains(Flags::PLACEMENT) {
                    continue 'siblings;
                }
                if let Some(state_node) = current.state_node {
                    return Some(state_node);
                }
                match current.first_child {
                    Some(child) => node = child,
                    None => continue 'siblings,
                }
            }
        }
    }

    /// Whether a fiber's node is the host parent an insertion targets.
    fn is_host_parent(&self, fiber: FiberId) -> bool {
        self.arena
            .get(fiber)
            .is_none_or(|fiber| fiber.tag == FiberTag::Root || fiber.state_node.is_some())
    }

    /// Removes a subtree from the host and frees its fibers.
    fn unmount(&mut self, fiber: FiberId, host_parent: Option<H::Node>) {
        let Some((own, alternate, first_child)) = self
            .arena
            .get(fiber)
            .map(|node| (node.state_node, node.alternate, node.first_child))
        else {
            return;
        };
        let children_parent = own.or(host_parent);

        // A deleted host fiber detaches its ref before its children are
        // removed, so the callback still sees the subtree whole. React keeps
        // the same order in `commitDeletionEffectsOnFiber`.
        self.detach_ref(fiber);

        let mut child = first_child;
        while let Some(current) = child {
            child = self
                .arena
                .get(current)
                .and_then(|fiber| fiber.sibling);
            self.unmount(current, children_parent);
        }

        if let Some(node) = self.arena.get(fiber) {
            // Detach first: a cleanup is user code and may still hold a state
            // handle, and an update from it must not name a fiber that is about
            // to leave the arena.
            node.hooks.detach();
            node.hooks.destroy_effects();
        }
        if let (Some(parent), Some(child)) = (host_parent, own) {
            self.host.remove(parent, child);
        }
        self.arena.take(fiber);
        if let Some(alternate) = alternate {
            self.arena.take(alternate);
        }
    }

    /// Runs the reference of a committed host fiber before its node goes away.
    ///
    /// A cleanup the callback returned on attach replaces the `None` call.
    /// Taking it out of the shared cell empties it for both alternates, so the
    /// cleanup cannot run twice — the rule React enforces by clearing
    /// `refCleanup` on the fiber and its alternate.
    fn detach_ref(&mut self, fiber: FiberId) {
        let Some(node) = self.arena.get(fiber) else {
            return;
        };
        let Some(reference) = node.memoized.reference().cloned() else {
            return;
        };
        let cleanup = node
            .ref_cleanup
            .as_ref()
            .and_then(|cleanup| cleanup.borrow_mut().take());
        match cleanup {
            Some(cleanup) => cleanup(),
            None => {
                reference(None);
            }
        }
    }

    /// Hands `order` the fibers of the subtree, children before parents.
    fn collect(&self, fiber: FiberId, order: &mut Vec<FiberId>) {
        let Some(node) = self.arena.get(fiber) else {
            return;
        };
        let mut child = node.first_child;
        while let Some(current) = child {
            self.collect(current, order);
            child = self
                .arena
                .get(current)
                .and_then(|fiber| fiber.sibling);
        }
        order.push(fiber);
    }

    /// Runs the cleanups of the previous runtime for the effects of `tag`.
    ///
    /// The records stay on the fiber: the matching phase creates their
    /// replacements later in the same commit.
    fn destroy_effects(&mut self, fiber: FiberId, tag: EffectTag) {
        let Some(node) = self.arena.get(fiber) else {
            return;
        };
        for effect in &node.effects {
            if effect.tag() == tag {
                effect.unmount();
            }
        }
    }

    /// Runs the effects of one fiber that belong to `tag` and retires them.
    ///
    /// The records of the other tag stay on the fiber for its own phase.
    fn create_effects(&mut self, fiber: FiberId, tag: EffectTag) {
        let effects = match self.arena.get_mut(fiber) {
            Some(node) => mem::take(&mut node.effects),
            None => return,
        };
        if effects.is_empty() {
            return;
        }
        let mut keep = Vec::new();
        for effect in effects {
            if effect.tag() == tag {
                effect.create();
            } else {
                keep.push(effect);
            }
        }
        if !keep.is_empty()
            && let Some(node) = self.arena.get_mut(fiber)
        {
            node.effects = keep;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Element;
    use crate::fiber::Fiber;
    use crate::hooks::{Cleanup, RenderCtx, mailbox};
    use crate::lane::Lanes;
    use crate::reconcile::reconcile_children;
    use crate::work::WorkLoop;
    use alloc::rc::Rc;
    use alloc::string::String;
    use alloc::vec;
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

    /// Drives a render and a commit over [`TreeHost`].
    struct Harness {
        host: TreeHost,
        arena: Arena<TreeHost>,
        mailbox: crate::hooks::MailboxHandle,
        ids: u64,
        root: FiberId,
    }

    impl Harness {
        fn new() -> Self {
            let host = TreeHost::new();
            let container = host.container();
            let mut arena = Arena::new();
            let mut root = Fiber::new(FiberTag::Root);
            root.state_node = Some(container);
            let root = arena.alloc(root);
            Self {
                host,
                arena,
                mailbox: mailbox(),
                ids: 0,
                root,
            }
        }

        fn render(&mut self, element: Element<TreeHost>) {
            // Rendering the root's element marks the root, the way
            // `Root::render` marks it with the default lane; without the mark
            // the root would bail out and the new element would never render.
            if let Some(root) = self.arena.get_mut(self.root) {
                root.lanes.insert(Lanes::SYNC);
            }
            let Some(work) = self.arena.create_work_in_progress(self.root) else {
                return;
            };
            let current_first = self
                .arena
                .get(work)
                .and_then(|fiber| fiber.alternate)
                .and_then(|alternate| self.arena.get(alternate))
                .and_then(|fiber| fiber.first_child);
            reconcile_children(&self.host, &mut self.arena, work, current_first, &[element]);
            let mut work_loop = WorkLoop::new(
                &mut self.host,
                &mut self.arena,
                Rc::clone(&self.mailbox),
                &mut self.ids,
                Lanes::SYNC,
            );
            work_loop.run(work);
            let mut commit = Commit::new(&mut self.host, &mut self.arena);
            commit.mutation(work);
            commit.layout(work);
            commit.passive(work);
            self.root = work;
        }

        fn children(&self) -> &[NodeId] {
            self.host.children(self.host.container())
        }
    }

    /// Every fiber reachable from the finished root, through children and
    /// siblings.
    fn fibers(harness: &Harness) -> Vec<FiberId> {
        let mut order = Vec::new();
        let mut pending = vec![harness.root];
        while let Some(fiber) = pending.pop() {
            let Some(node) = harness.arena.get(fiber) else {
                continue;
            };
            order.push(fiber);
            pending.extend(node.first_child);
            pending.extend(node.sibling);
        }
        order
    }

    #[test]
    fn a_mount_attaches_the_host_nodes_in_order() {
        let mut harness = Harness::new();
        harness.render(Element::fragment([
            Element::host(1),
            Element::host(2),
            Element::host(3),
        ]));
        let children = harness.children().to_vec();
        assert_eq!(children.len(), 3);
        assert_eq!(
            children
                .iter()
                .map(|node| harness.host.element(*node))
                .collect::<Vec<_>>(),
            [Some(1), Some(2), Some(3)]
        );
    }

    #[test]
    fn a_host_element_places_its_children_inside_its_node() {
        let mut harness = Harness::new();
        harness.render(Element::host(1).children([Element::host(2), Element::host(3)]));
        let outer = harness.children().to_vec();
        assert_eq!(outer.len(), 1);
        assert_eq!(harness.host.element(outer[0]), Some(1));
        let inner = harness.host.children(outer[0]).to_vec();
        assert_eq!(
            inner
                .iter()
                .map(|node| harness.host.element(*node))
                .collect::<Vec<_>>(),
            [Some(2), Some(3)]
        );
    }

    #[test]
    fn a_reorder_moves_the_nodes_instead_of_recreating_them() {
        let mut harness = Harness::new();
        harness.render(Element::fragment([
            Element::host(1).keyed("a"),
            Element::host(2).keyed("b"),
            Element::host(3).keyed("c"),
        ]));
        let before = harness.children().to_vec();
        assert_eq!(before.len(), 3);

        harness.render(Element::fragment([
            Element::host(3).keyed("c"),
            Element::host(1).keyed("a"),
            Element::host(2).keyed("b"),
        ]));
        let after = harness.children().to_vec();
        assert_eq!(after, [before[2], before[0], before[1]]);
        assert_eq!(
            after
                .iter()
                .map(|node| harness.host.element(*node))
                .collect::<Vec<_>>(),
            [Some(3), Some(1), Some(2)]
        );
    }

    #[test]
    fn a_removed_child_leaves_the_host_and_is_not_recreated() {
        let mut harness = Harness::new();
        harness.render(Element::fragment([Element::host(1), Element::host(2)]));
        let kept = harness.children()[0];

        harness.render(Element::fragment([Element::host(1)]));
        assert_eq!(harness.children(), [kept]);
        assert_eq!(harness.host.element(kept), Some(1));

        let created = harness.host.nodes.len();
        harness.render(Element::fragment([Element::host(1)]));
        assert_eq!(harness.host.nodes.len(), created);
    }

    #[test]
    fn a_layout_effect_runs_on_mount_and_cleans_up_on_unmount() {
        let log = Rc::new(RefCell::new(Vec::<&'static str>::new()));
        let element = {
            let log = Rc::clone(&log);
            Element::component(move |ctx: &mut RenderCtx<'_>| {
                let run = Rc::clone(&log);
                let clean = Rc::clone(&log);
                ctx.use_layout_effect((), move || {
                    run.borrow_mut().push("run");
                    let clean = Rc::clone(&clean);
                    Some(Box::new(move || {
                        clean.borrow_mut().push("cleanup");
                    }) as Cleanup)
                });
                Element::host(1)
            })
        };

        let mut harness = Harness::new();
        harness.render(element);
        assert_eq!(*log.borrow(), ["run"]);

        harness.render(Element::empty());
        assert_eq!(*log.borrow(), ["run", "cleanup"]);
        assert!(harness.children().is_empty());
    }

    #[test]
    fn a_child_layout_effect_runs_before_its_parent() {
        let log = Rc::new(RefCell::new(Vec::<String>::new()));
        let child = {
            let log = Rc::clone(&log);
            Element::component(move |ctx: &mut RenderCtx<'_>| {
                let log = Rc::clone(&log);
                ctx.use_layout_effect((), move || {
                    log.borrow_mut().push(String::from("child"));
                    None
                });
                Element::empty()
            })
        };
        let parent = {
            let log = Rc::clone(&log);
            Element::component(move |ctx: &mut RenderCtx<'_>| {
                let log = Rc::clone(&log);
                ctx.use_layout_effect((), move || {
                    log.borrow_mut().push(String::from("parent"));
                    None
                });
                child.clone()
            })
        };

        let mut harness = Harness::new();
        harness.render(parent);
        assert_eq!(*log.borrow(), [String::from("child"), String::from("parent")]);
    }

    /// A component that logs its layout create and cleanup whenever the epoch
    /// changes, and renders `child`.
    fn logging_component(
        log: &Rc<RefCell<Vec<&'static str>>>,
        epoch: &Rc<Cell<u32>>,
        create: &'static str,
        cleanup: &'static str,
        child: Element<TreeHost>,
    ) -> Element<TreeHost> {
        let log = Rc::clone(log);
        let epoch = Rc::clone(epoch);
        Element::component(move |ctx: &mut RenderCtx<'_>| {
            let run = Rc::clone(&log);
            let destroy = Rc::clone(&log);
            ctx.use_layout_effect((epoch.get(),), move || {
                run.borrow_mut().push(create);
                let destroy = Rc::clone(&destroy);
                Some(Box::new(move || {
                    destroy.borrow_mut().push(cleanup);
                }) as Cleanup)
            });
            child.clone()
        })
    }

    /// A component with a layout and a passive effect that re-run whenever the
    /// epoch changes, rendering host `9`.
    fn epoch_probe(epoch: &Rc<Cell<u32>>) -> Element<TreeHost> {
        let epoch = Rc::clone(epoch);
        Element::component(move |ctx: &mut RenderCtx<'_>| {
            let tick = epoch.get();
            ctx.use_layout_effect((tick,), || None);
            ctx.use_effect((tick,), || None);
            Element::host(9)
        })
    }

    #[test]
    fn every_layout_cleanup_runs_before_any_layout_create() {
        let log = Rc::new(RefCell::new(Vec::<&'static str>::new()));
        let epoch = Rc::new(Cell::new(0_u32));
        let element = || {
            let child = logging_component(&log, &epoch, "child create", "child cleanup", Element::empty());
            logging_component(&log, &epoch, "parent create", "parent cleanup", child)
        };

        let mut harness = Harness::new();
        harness.render(element());
        assert_eq!(*log.borrow(), ["child create", "parent create"]);

        log.borrow_mut().clear();
        epoch.set(1);
        harness.render(element());
        assert_eq!(
            *log.borrow(),
            ["child cleanup", "parent cleanup", "child create", "parent create"]
        );
    }

    #[test]
    fn a_passive_effect_runs_after_the_layout_phase() {
        let log = Rc::new(RefCell::new(Vec::<&'static str>::new()));
        let element = {
            let log = Rc::clone(&log);
            Element::component(move |ctx: &mut RenderCtx<'_>| {
                let layout = Rc::clone(&log);
                ctx.use_layout_effect((), move || {
                    layout.borrow_mut().push("layout");
                    None
                });
                let passive = Rc::clone(&log);
                ctx.use_effect((), move || {
                    passive.borrow_mut().push("passive");
                    None
                });
                Element::empty()
            })
        };

        let mut harness = Harness::new();
        harness.render(element);
        assert_eq!(*log.borrow(), ["layout", "passive"]);
    }

    #[test]
    fn the_commit_leaves_no_flag_of_any_phase_behind() {
        let epoch = Rc::new(Cell::new(0_u32));

        let mut harness = Harness::new();
        harness.render(Element::fragment([
            Element::host(1).keyed("a"),
            epoch_probe(&epoch),
            Element::host(2).keyed("b"),
        ]));
        assert_eq!(harness.children().len(), 3);

        // A reorder, an insertion and a component whose effect deps changed:
        // every phase of the commit has work to do and flags to consume.
        // `epoch_probe` builds a fresh element so the component renders again;
        // a reused element would bail out and leave both effect phases empty.
        epoch.set(1);
        harness.render(Element::fragment([
            Element::host(2).keyed("b"),
            Element::host(1).keyed("a"),
            epoch_probe(&epoch),
            Element::host(3).keyed("c"),
        ]));
        assert_eq!(harness.children().len(), 4);

        let order = fibers(&harness);
        let component = order.iter().any(|fiber| {
            harness
                .arena
                .get(*fiber)
                .is_some_and(|node| node.tag == FiberTag::Component)
        });
        assert!(component, "the walk must reach the component fiber");
        for fiber in order {
            let Some(node) = harness.arena.get(fiber) else {
                continue;
            };
            // The finished tree shares fibers with the committed one, so a
            // flag that outlived its phase would replay it in a later commit.
            assert!(
                node.flags.is_empty(),
                "stale {:?} left on {:?}",
                node.flags,
                node.tag
            );
        }
    }
}
