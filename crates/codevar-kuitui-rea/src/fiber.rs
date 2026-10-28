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

//! The double-buffered fiber arena.
//!
//! A fiber is a node of the tree the reconciler works on. It is stored in one
//! arena and addressed by [`FiberId`], never by reference, so a parent can
//! point at a child and vice versa without fighting the borrow checker.
//!
//! # Double buffering
//!
//! Every fiber has an *alternate*: the slot on the other tree. The committed
//! tree is what the host currently shows; a render builds a work-in-progress
//! tree beside it, reusing each fiber's alternate. Nothing the render does can
//! be observed on screen, so an interrupted render is thrown away by pointing
//! the root back at the committed tree — the alternate is already there and
//! keeps its own flags and payloads.
//!
//! [`Arena::create_work_in_progress`] is the only way to start a node for a
//! render. It copies the committed payload into the pending slot, resets the
//! per-render bookkeeping (flags, deletions) and returns the alternate. The
//! parent's reconciliation then overwrites the pending payload if the node's
//! props changed.
//!
//! # Throwing a render away
//!
//! A render can be interrupted before it commits. The tree it built never
//! reaches the host, but the arena owns the fibers it allocated and the hook
//! queues it replayed. [`Arena::begin_render`] starts recording every fiber a
//! render starts; [`Arena::abandon_render`] frees what the render allocated,
//! returns each queue to the fiber that owns it in the committed tree and puts
//! the queue back to the state the last commit left. [`Arena::end_render`]
//! stops recording once the tree is complete, before the commit runs.
//!
//! # Why the tree is not a `Vec<Fiber>`
//!
//! Fibers are both created and destroyed between commits (a conditional
//! branch, a list item removed), so the arena keeps a free list and recycles
//! slots. The two halves of an alternate are freed together: the slot of one
//! half must never be handed out while the other half still points at it.

// The work loop that consumes these primitives lives in `work`.
#![allow(dead_code)]

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::any::TypeId;
use core::fmt;
use core::ops::{BitAnd, BitOr, Not};

use crate::element::{ComponentFn, Element, HostNode};
use crate::hooks::{EffectRecord, HookList};
use crate::host::{Host, Key};
use crate::lane::Lanes;

/// Bookkeeping the commit phase needs, attached to a fiber.
///
/// Flags are per-alternate. Starting a work-in-progress node resets them (see
/// [`Fiber::work_in_progress`]), and every commit phase also clears the mask it
/// applied. The second rule is what keeps a finished tree honest: a fiber can
/// be shared with the committed tree instead of cloned, and a flag left on it
/// would make a later commit redo work that already ran. The masks together
/// cover [`Flags::ALL`].
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Flags(u16);

impl Flags {
    /// Nothing to do.
    pub(crate) const NONE: Self = Self(0);

    /// The node is new in this tree and must be inserted into the host.
    pub(crate) const PLACEMENT: Self = Self(1 << 0);

    /// The node's host description changed and must be updated.
    pub(crate) const UPDATE: Self = Self(1 << 1);

    /// The node's `deletions` list must be processed before it is mutated.
    pub(crate) const CHILD_DELETION: Self = Self(1 << 2);

    /// The node's children must be removed because its content changed kind.
    pub(crate) const CONTENT_RESET: Self = Self(1 << 3);

    /// The node has a ref to attach or detach.
    pub(crate) const REF: Self = Self(1 << 4);

    /// The node has a callback ref to invoke.
    pub(crate) const CALLBACK: Self = Self(1 << 5);

    /// The node has a layout effect to run.
    pub(crate) const LAYOUT: Self = Self(1 << 6);

    /// The node has a passive effect to run.
    pub(crate) const PASSIVE: Self = Self(1 << 7);

    /// The node's effect cleanup must run before its parent is mutated.
    pub(crate) const INSERTION: Self = Self(1 << 8);

    /// Every flag.
    pub(crate) const ALL: Self = Self((1 << 9) - 1);

    /// The flags the mutation phase applies and consumes.
    pub(crate) const MUTATION_MASK: Self = Self(
        Self::PLACEMENT.0
            | Self::UPDATE.0
            | Self::CHILD_DELETION.0
            | Self::CONTENT_RESET.0
            | Self::REF.0
            | Self::CALLBACK.0
            | Self::INSERTION.0,
    );

    /// The flags the layout phase applies and consumes.
    pub(crate) const LAYOUT_MASK: Self =
        Self(Self::UPDATE.0 | Self::CALLBACK.0 | Self::REF.0 | Self::LAYOUT.0);

    /// The flags the passive phase applies and consumes.
    pub(crate) const PASSIVE_MASK: Self = Self(Self::PASSIVE.0);

    /// Whether no flag is set.
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every flag of `other` is set.
    pub(crate) const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether any flag of `other` is set.
    pub(crate) const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Adds the flags of `other`.
    pub(crate) const fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Removes the flags of `other`.
    pub(crate) const fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }
}

impl BitOr for Flags {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitAnd for Flags {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
}

impl Not for Flags {
    type Output = Self;

    fn not(self) -> Self {
        Self(!self.0 & Self::ALL.0)
    }
}

impl fmt::Debug for Flags {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Flags({:#b})", self.0)
    }
}

/// What a fiber renders.
///
/// The payload is shared with the element that produced it, so starting a
/// work-in-progress node never copies a host description or a prop bag.
pub(crate) enum Payload<H: Host> {
    /// A node with no content of its own, such as the root.
    None,
    /// The render function of a component.
    Component(Rc<ComponentFn<H>>),
    /// A host element together with the children it arranges, shared with the
    /// committed tree until the element changes.
    ///
    /// The whole description is the payload, not just the node: an element's
    /// identity is what decides whether a fiber can keep its subtree, and the
    /// children are part of that description.
    Host(Rc<HostNode<H>>),
    /// The children a fragment describes.
    Fragment(Rc<[Element<H>]>),
}

impl<H: Host> Clone for Payload<H> {
    fn clone(&self) -> Self {
        match self {
            Self::None => Self::None,
            Self::Component(render) => Self::Component(Rc::clone(render)),
            Self::Host(node) => Self::Host(Rc::clone(node)),
            Self::Fragment(children) => Self::Fragment(Rc::clone(children)),
        }
    }
}

impl<H: Host> Payload<H> {
    /// Whether two payloads describe the same element.
    ///
    /// Identity, not equality: a render that hands the fiber the same `Rc` is
    /// a render that described the same thing, which is the check React makes
    /// on `memoizedProps` vs `pendingProps`. Only then can the fiber keep the
    /// subtree it rendered last time.
    pub(crate) fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::None, Self::None) => true,
            (Self::Component(a), Self::Component(b)) => Rc::ptr_eq(a, b),
            (Self::Host(a), Self::Host(b)) => Rc::ptr_eq(a, b),
            (Self::Fragment(a), Self::Fragment(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// What kind of node a fiber is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FiberTag {
    /// The root of a tree. Its `state_node` is the host container.
    Root,
    /// A component; its children are whatever the component returned.
    Component,
    /// A host element.
    Host,
    /// A transparent node around a list of children.
    Fragment,
}

/// An index into the fiber arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FiberId(u32);

impl FiberId {
    /// The index of the fiber in the arena.
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }

    /// Builds an id from an arena index.
    pub(crate) const fn from_index(index: u32) -> Self {
        Self(index)
    }
}

/// A node of the tree the reconciler works on.
pub(crate) struct Fiber<H: Host> {
    /// What kind of node this is.
    pub(crate) tag: FiberTag,
    /// The identity of the node among its siblings, when it has one.
    pub(crate) key: Option<Key>,
    /// The component identity, used to decide whether a node can be reused.
    pub(crate) element_type: Option<TypeId>,
    /// The payload the current render produced.
    pub(crate) pending: Payload<H>,
    /// The payload the last commit applied.
    pub(crate) memoized: Payload<H>,
    /// The parent, or `None` for the root.
    pub(crate) parent: Option<FiberId>,
    /// The first child.
    pub(crate) first_child: Option<FiberId>,
    /// The next sibling.
    pub(crate) sibling: Option<FiberId>,
    /// The position among the siblings, as last reconciled.
    pub(crate) index: usize,
    /// The host node, shared by both alternates.
    pub(crate) state_node: Option<H::Node>,
    /// The hooks of a component.
    pub(crate) hooks: HookList,
    /// The effects this render must commit.
    pub(crate) effects: Vec<EffectRecord>,
    /// What must be done to this node in the commit.
    pub(crate) flags: Flags,
    /// What must be done to the subtree, including this node.
    pub(crate) subtree_flags: Flags,
    /// The children that were deleted from this node and must be removed from
    /// the host before anything is inserted.
    pub(crate) deletions: Vec<FiberId>,
    /// The lanes of updates that landed on this node.
    pub(crate) lanes: Lanes,
    /// The lanes of updates anywhere below this node.
    pub(crate) child_lanes: Lanes,
    /// The node on the other tree.
    pub(crate) alternate: Option<FiberId>,
}

impl<H: Host> Fiber<H> {
    /// Creates an empty fiber of the given kind.
    pub(crate) fn new(tag: FiberTag) -> Self {
        Self {
            tag,
            key: None,
            element_type: None,
            pending: Payload::None,
            memoized: Payload::None,
            parent: None,
            first_child: None,
            sibling: None,
            index: 0,
            state_node: None,
            hooks: HookList::default(),
            effects: Vec::new(),
            flags: Flags::NONE,
            subtree_flags: Flags::NONE,
            deletions: Vec::new(),
            lanes: Lanes::NONE,
            child_lanes: Lanes::NONE,
            alternate: None,
        }
    }

    /// Creates the work-in-progress copy of a committed fiber.
    ///
    /// The committed payload is copied into both the pending and the memoized
    /// slot, so a node whose parent bails out keeps rendering what is on
    /// screen. The per-render bookkeeping starts empty: the commit flags, the
    /// deletions and the sibling link belong to the render that is about to
    /// start. Inheriting the old sibling would splice the committed list into
    /// the new one, so reconciliation relinks the children from scratch.
    pub(crate) fn work_in_progress(current: &Self) -> Self {
        Self {
            tag: current.tag,
            key: current.key.clone(),
            element_type: current.element_type,
            pending: current.memoized.clone(),
            memoized: current.memoized.clone(),
            parent: current.parent,
            first_child: current.first_child,
            sibling: None,
            index: current.index,
            state_node: current.state_node,
            hooks: current.hooks.clone(),
            effects: Vec::new(),
            flags: Flags::NONE,
            subtree_flags: Flags::NONE,
            deletions: Vec::new(),
            lanes: current.lanes,
            child_lanes: current.child_lanes,
            alternate: None,
        }
    }

    /// Records a child that must be removed from the host.
    pub(crate) fn delete_child(&mut self, child: FiberId) {
        self.flags.insert(Flags::CHILD_DELETION);
        self.deletions.push(child);
    }
}

/// A fiber a render started.
struct Started {
    /// The work-in-progress fiber.
    id: FiberId,
    /// Whether the render allocated the slot, so throwing the render away frees
    /// it. A reused alternate keeps its slot for the next render to overwrite.
    allocated: bool,
}

/// The arena that owns every fiber.
pub(crate) struct Arena<H: Host> {
    /// The slots. `None` marks a slot on the free list.
    fibers: Vec<Option<Fiber<H>>>,
    /// The indices of the free slots.
    free: Vec<u32>,
    /// The fibers the render in flight started, in the order it started them.
    started: Option<Vec<Started>>,
}

impl<H: Host> Arena<H> {
    /// Creates an empty arena.
    pub(crate) fn new() -> Self {
        Self {
            fibers: Vec::new(),
            free: Vec::new(),
            started: None,
        }
    }

    /// The number of live fibers.
    pub(crate) fn len(&self) -> usize {
        self.fibers
            .iter()
            .filter(|slot| slot.is_some())
            .count()
    }

    /// The fiber at `id`, if the slot is live.
    pub(crate) fn get(&self, id: FiberId) -> Option<&Fiber<H>> {
        self.fibers
            .get(id.index())
            .and_then(Option::as_ref)
    }

    /// The fiber at `id` mutably, if the slot is live.
    pub(crate) fn get_mut(&mut self, id: FiberId) -> Option<&mut Fiber<H>> {
        self.fibers
            .get_mut(id.index())
            .and_then(Option::as_mut)
    }

    /// Stores a fiber and returns its id, recycling a free slot when possible.
    pub(crate) fn alloc(&mut self, fiber: Fiber<H>) -> FiberId {
        let id = if let Some(index) = self.free.pop() {
            if let Some(slot) = self.fibers.get_mut(index as usize) {
                *slot = Some(fiber);
            }
            FiberId(index)
        } else {
            let index = self.fibers.len() as u32;
            self.fibers.push(Some(fiber));
            FiberId(index)
        };
        if let Some(started) = &mut self.started {
            started.push(Started { id, allocated: true });
        }
        id
    }

    /// Removes a fiber from the arena and reclaims its slot.
    ///
    /// The other half of the alternate is detached but not freed, because the
    /// caller may still need its host node to remove it from the tree.
    pub(crate) fn take(&mut self, id: FiberId) -> Option<Fiber<H>> {
        let slot = self.fibers.get_mut(id.index())?;
        let fiber = slot.take()?;
        self.free.push(id.0);
        if let Some(alternate) = fiber.alternate
            && let Some(other) = self.get_mut(alternate)
        {
            other.alternate = None;
        }
        Some(fiber)
    }

    /// Returns the alternate of `current`, or creates it.
    ///
    /// The committed fiber and its alternate share one host node; only one of
    /// them is reachable from the root at a time.
    pub(crate) fn create_work_in_progress(&mut self, current: FiberId) -> Option<FiberId> {
        let mut fiber = Fiber::work_in_progress(self.get(current)?);
        if let Some(alternate) = self
            .get(current)
            .and_then(|fiber| fiber.alternate)
        {
            fiber.alternate = Some(current);
            if let Some(slot) = self.fibers.get_mut(alternate.index())
                && slot.is_some()
            {
                *slot = Some(fiber);
                if let Some(started) = &mut self.started {
                    started.push(Started {
                        id: alternate,
                        allocated: false,
                    });
                }
                return Some(alternate);
            }
        }
        let id = self.alloc(fiber);
        if let Some(fiber) = self.get_mut(id) {
            fiber.alternate = Some(current);
        }
        if let Some(fiber) = self.get_mut(current) {
            fiber.alternate = Some(id);
        }
        Some(id)
    }

    /// Starts recording the fibers a render starts.
    ///
    /// Only a render that can be interrupted needs the record; the work loop
    /// calls this when it prepares a fresh tree.
    pub(crate) fn begin_render(&mut self) {
        debug_assert!(
            self.started.is_none(),
            "a render cannot start before the previous one ended"
        );
        self.started = Some(Vec::new());
    }

    /// Stops recording the fibers a render starts.
    ///
    /// The tree is complete at this point, so it is about to become the
    /// committed one and there is nothing left to throw away.
    pub(crate) fn end_render(&mut self) {
        self.started = None;
    }

    /// Throws away the tree the render in flight built.
    ///
    /// Every fiber the render started is detached from its update queues: the
    /// queues return to the fiber that owns them in the committed tree, because
    /// a state handle must not wake a fiber the render abandoned — and a freed
    /// slot is handed out again. The queue itself goes back to the state the
    /// last commit left, so the next render can replay the updates this one
    /// consumed. The fibers the render allocated are freed; a reused alternate
    /// keeps its slot, which the next render overwrites from the committed
    /// fiber.
    pub(crate) fn abandon_render(&mut self) {
        let Some(started) = self.started.take() else {
            return;
        };
        for started in started.into_iter().rev() {
            let Some(fiber) = self.get(started.id) else {
                continue;
            };
            fiber.hooks.restore_committed();
            match fiber.alternate {
                Some(owner) => fiber.hooks.reset_owners(owner),
                None => fiber.hooks.detach(),
            }
            if started.allocated {
                self.take(started.id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct TestNode(u32);

    struct TestHost {
        nodes: Vec<u32>,
    }

    impl Host for TestHost {
        type Element = u32;
        type Node = TestNode;

        fn element_matches(&self, old: &u32, new: &u32) -> bool {
            old == new
        }

        fn create(&mut self, element: &u32) -> TestNode {
            self.nodes.push(*element);
            TestNode(self.nodes.len() as u32)
        }

        fn update(&mut self, _node: TestNode, _element: &u32) {}

        fn insert_before(&mut self, _parent: TestNode, _child: TestNode, _anchor: Option<TestNode>) {}

        fn remove(&mut self, _parent: TestNode, _child: TestNode) {}
    }

    fn arena() -> Arena<TestHost> {
        Arena::new()
    }

    fn fiber(tag: FiberTag) -> Fiber<TestHost> {
        Fiber::new(tag)
    }

    fn component_payload() -> Payload<TestHost> {
        Payload::Component(Rc::new(|_| Element::empty()))
    }

    #[test]
    fn allocation_recycles_freed_slots() {
        let mut arena = arena();
        let first = arena.alloc(fiber(FiberTag::Host));
        arena.alloc(fiber(FiberTag::Host));
        assert_eq!(arena.len(), 2);
        assert!(arena.take(first).is_some());
        assert_eq!(arena.len(), 1);
        let third = arena.alloc(fiber(FiberTag::Fragment));
        assert_eq!(third, first);
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.get(third).map(|fiber| fiber.tag), Some(FiberTag::Fragment));
    }

    #[test]
    fn taking_a_fiber_detaches_its_alternate() {
        let mut arena = arena();
        let current = arena.alloc(fiber(FiberTag::Component));
        let work = arena
            .create_work_in_progress(current)
            .unwrap_or(current);
        assert_eq!(
            arena
                .get(current)
                .and_then(|fiber| fiber.alternate),
            Some(work)
        );
        assert!(arena.take(current).is_some());
        assert_eq!(arena.get(work).and_then(|fiber| fiber.alternate), None);
    }

    #[test]
    fn work_in_progress_reuses_the_alternate() {
        let mut arena = arena();
        let mut current = fiber(FiberTag::Component);
        current.element_type = Some(TypeId::of::<u32>());
        current.pending = component_payload();
        current.memoized = current.pending.clone();
        current.flags = Flags::UPDATE | Flags::PLACEMENT;
        current.deletions.push(FiberId(7));
        let current = arena.alloc(current);

        let first = arena.create_work_in_progress(current);
        let second = arena.create_work_in_progress(current);
        assert_eq!(first, second);
        assert_eq!(arena.len(), 2);

        let work = first.and_then(|id| arena.get(id));
        assert!(work.is_some());
        let Some(work) = work else { return };
        assert!(work.flags.is_empty());
        assert_eq!(work.subtree_flags, Flags::NONE);
        assert!(work.deletions.is_empty());
        assert_eq!(work.element_type, Some(TypeId::of::<u32>()));
        assert_eq!(work.alternate, Some(current));
        assert!(matches!(work.pending, Payload::Component(_)));
    }

    #[test]
    fn abandoning_a_render_frees_what_it_allocated() {
        let mut arena = arena();
        let committed = arena.alloc(fiber(FiberTag::Component));
        arena.begin_render();
        let Some(work) = arena.create_work_in_progress(committed) else {
            return;
        };
        // A fiber the render created from scratch, the way reconciliation
        // allocates a child with no committed counterpart.
        let created = arena.alloc(fiber(FiberTag::Host));
        assert_eq!(arena.len(), 3);

        arena.abandon_render();
        assert_eq!(arena.len(), 1);
        assert!(arena.get(work).is_none());
        assert!(arena.get(created).is_none());
        assert!(arena.get(committed).is_some());
        // The committed fiber no longer points at the freed half.
        assert_eq!(
            arena
                .get(committed)
                .and_then(|fiber| fiber.alternate),
            None
        );
    }

    #[test]
    fn abandoning_a_render_keeps_a_reused_alternate() {
        let mut arena = arena();
        let committed = arena.alloc(fiber(FiberTag::Component));
        let Some(alternate) = arena.create_work_in_progress(committed) else {
            return;
        };

        // The slot an earlier render allocated is reused, not allocated again.
        arena.begin_render();
        assert_eq!(arena.create_work_in_progress(committed), Some(alternate));
        arena.abandon_render();

        assert_eq!(arena.len(), 2);
        assert_eq!(
            arena
                .get(committed)
                .and_then(|fiber| fiber.alternate),
            Some(alternate)
        );
        assert_eq!(
            arena
                .get(alternate)
                .and_then(|fiber| fiber.alternate),
            Some(committed)
        );
    }

    #[test]
    fn a_completed_render_stops_recording() {
        let mut arena = arena();
        arena.alloc(fiber(FiberTag::Component));
        arena.begin_render();
        arena.end_render();
        // The tree committed: a later abandon has nothing to throw away.
        let work = arena.alloc(fiber(FiberTag::Host));
        arena.abandon_render();
        assert_eq!(arena.len(), 2);
        assert!(arena.get(work).is_some());
    }

    #[test]
    fn work_in_progress_copies_children_and_lanes() {
        let mut arena = arena();
        let mut current = fiber(FiberTag::Host);
        let child = arena.alloc(fiber(FiberTag::Host));
        current.first_child = Some(child);
        current.child_lanes = Lanes::DEFAULT;
        current.lanes = Lanes::SYNC;
        current.state_node = Some(TestNode(3));
        let current = arena.alloc(current);

        let Some(work) = arena
            .create_work_in_progress(current)
            .and_then(|id| arena.get(id))
        else {
            return;
        };
        assert_eq!(work.first_child, Some(child));
        assert_eq!(work.child_lanes, Lanes::DEFAULT);
        assert_eq!(work.lanes, Lanes::SYNC);
        assert_eq!(work.state_node, Some(TestNode(3)));
        assert!(matches!(work.pending, Payload::None));
    }

    #[test]
    fn deleting_a_child_marks_the_parent() {
        let mut fiber = fiber(FiberTag::Host);
        fiber.delete_child(FiberId(4));
        assert!(fiber.flags.contains(Flags::CHILD_DELETION));
        assert_eq!(fiber.deletions, [FiberId(4)]);
    }

    #[test]
    fn a_payload_is_shared_between_alternates() {
        let element: Rc<HostNode<TestHost>> = Rc::new(HostNode {
            element: Rc::new(7),
            children: Rc::from(Vec::<Element<TestHost>>::new()),
        });
        let payload: Payload<TestHost> = Payload::Host(Rc::clone(&element));
        let clone = payload.clone();
        let (Payload::Host(a), Payload::Host(b)) = (payload, clone) else {
            return;
        };
        assert!(Rc::ptr_eq(&a, &b));
    }

    #[test]
    fn the_phase_masks_cover_every_flag() {
        let masks = Flags::MUTATION_MASK | Flags::LAYOUT_MASK | Flags::PASSIVE_MASK;
        // A flag outside every mask would survive the commit that applied it.
        assert!(masks.contains(Flags::ALL));
    }
}
