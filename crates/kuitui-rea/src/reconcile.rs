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

//! Keyed reconciliation.
//!
//! [`reconcile_children`] diffs the children the render just produced against
//! the children of the committed tree and decides, for every new child,
//! whether it reuses an existing fiber or needs a new one. The result is a
//! work-in-progress child list plus, on the parent, the fibers that must be
//! removed from the host at commit.
//!
//! # Two passes
//!
//! The first pass walks both lists in lockstep and reuses every child whose
//! key and type still match at the same position. That is the case a normal
//! render produces, and it costs one comparison per child.
//!
//! The second pass files the remaining old children in a map, keyed by their
//! key or, when they have none, by the position they occupy, and looks up the
//! remaining new children in it. A reorder or an insertion therefore finds its
//! old fiber without scanning, and the whole diff stays linear in the number
//! of children. Old children nobody claimed are deleted.
//!
//! # Placement
//!
//! Reusing a fiber is not enough: the commit phase has to know which children
//! moved, because moving a node is not free. A child whose old position is
//! already behind the last position the diff placed must be moved; everything
//! else keeps its order. This is React's `lastPlacedIndex` rule, and it is what
//! keeps a list rotation from touching every node.
//!
//! # The committed tree
//!
//! Reconciliation only writes to work-in-progress fibers. It reads the
//! committed children to find keys, indices and host descriptions, but it never
//! links, moves or deletes them, so an interrupted render leaves the tree on
//! screen exactly as it was.

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use core::cmp::Ordering;

use crate::element::{Element, Kind};
use crate::fiber::{Arena, Fiber, FiberId, FiberTag, Flags, Payload};
use crate::host::{Host, Key};

/// Diffs `children` against the children of the committed tree.
///
/// `parent` is the work-in-progress parent and `current_first` its committed
/// first child. The function rewrites `parent`'s child list, records the
/// deletions on the parent and returns the first new child.
pub(crate) fn reconcile_children<H: Host>(
    host: &H,
    arena: &mut Arena<H>,
    parent: FiberId,
    current_first: Option<FiberId>,
    children: &[Element<H>],
) -> Option<FiberId> {
    let mut layout = Layout::default();
    let mut old = current_first;
    let mut position = 0;

    while position < children.len() {
        let element = &children[position];
        if element.is_empty() {
            position += 1;
            continue;
        }
        let Some(current) = old else { break };
        let matches = arena
            .get(current)
            .is_some_and(|fiber| fiber.key.as_ref() == element.key() && same_type(host, fiber, element));
        if !matches {
            break;
        }
        position += 1;
        old = arena.get(current).and_then(|fiber| fiber.sibling);
        let child = reuse(arena, current, element);
        layout.push(arena, parent, child);
    }

    let mut remaining = collect(arena, old);
    while position < children.len() {
        let element = &children[position];
        position += 1;
        if element.is_empty() {
            continue;
        }
        let lookup = match element.key() {
            Some(key) => MatchKey::Key(key.clone()),
            None => MatchKey::Index(layout.index),
        };
        let matched = remaining.get(&lookup).copied().filter(|current| {
            arena
                .get(*current)
                .is_some_and(|fiber| same_type(host, fiber, element))
        });
        let child = match matched {
            Some(current) => {
                remaining.remove(&lookup);
                reuse(arena, current, element)
            }
            None => create(arena, element),
        };
        layout.push(arena, parent, child);
    }

    for (_, current) in remaining {
        if let Some(fiber) = arena.get_mut(parent) {
            fiber.delete_child(current);
        }
    }

    if let Some(fiber) = arena.get_mut(parent) {
        fiber.first_child = layout.first;
    }
    layout.first
}

/// The first child of the committed half of a fiber.
pub(crate) fn committed_first_child<H: Host>(arena: &Arena<H>, fiber: FiberId) -> Option<FiberId> {
    arena
        .get(fiber)
        .and_then(|fiber| fiber.alternate)
        .and_then(|alternate| arena.get(alternate))
        .and_then(|fiber| fiber.first_child)
}

/// Builds the work-in-progress child list and decides what moved.
#[derive(Default)]
struct Layout {
    /// The first child of the new list.
    first: Option<FiberId>,
    /// The child the next one is linked after.
    previous: Option<FiberId>,
    /// The highest old position placed so far.
    last_placed: usize,
    /// The position of the next new child.
    index: usize,
}

impl Layout {
    /// Links `child` as the next child and flags it when it must move.
    fn push<H: Host>(&mut self, arena: &mut Arena<H>, parent: FiberId, child: FiberId) {
        if self.first.is_none() {
            self.first = Some(child);
        }
        if let Some(previous) = self.previous
            && let Some(fiber) = arena.get_mut(previous)
        {
            fiber.sibling = Some(child);
        }
        let old_index = arena
            .get(child)
            .and_then(|fiber| fiber.alternate)
            .and_then(|alternate| arena.get(alternate))
            .map(|fiber| fiber.index);
        if let Some(fiber) = arena.get_mut(child) {
            fiber.parent = Some(parent);
            fiber.index = self.index;
            match old_index {
                Some(old) if old >= self.last_placed => self.last_placed = old,
                _ => fiber.flags.insert(Flags::PLACEMENT),
            }
        }
        self.previous = Some(child);
        self.index += 1;
    }
}

/// The identity an old child is filed under for the second pass.
#[derive(PartialEq, Eq)]
enum MatchKey {
    /// A child that has a key.
    Key(Key),
    /// A child with no key, filed under the position it occupies.
    Index(usize),
}

impl Ord for MatchKey {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Key(a), Self::Key(b)) => a.as_str().cmp(b.as_str()),
            (Self::Key(_), Self::Index(_)) => Ordering::Less,
            (Self::Index(_), Self::Key(_)) => Ordering::Greater,
            (Self::Index(a), Self::Index(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for MatchKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Files the old children from `old` on in the map of the second pass.
fn collect<H: Host>(arena: &Arena<H>, mut old: Option<FiberId>) -> BTreeMap<MatchKey, FiberId> {
    let mut remaining = BTreeMap::new();
    while let Some(current) = old {
        old = arena.get(current).and_then(|fiber| fiber.sibling);
        if let Some(fiber) = arena.get(current) {
            let key = match &fiber.key {
                Some(key) => MatchKey::Key(key.clone()),
                None => MatchKey::Index(fiber.index),
            };
            remaining.insert(key, current);
        }
    }
    remaining
}

/// Whether the committed fiber can become the element's fiber again.
fn same_type<H: Host>(host: &H, fiber: &Fiber<H>, element: &Element<H>) -> bool {
    match (fiber.tag, element.kind()) {
        (FiberTag::Host, Kind::Host(new)) => match &fiber.memoized {
            Payload::Host(old) => host.element_matches(&old.element, &new.element),
            _ => false,
        },
        (FiberTag::Component, Kind::Component { type_id, .. }) => fiber.element_type == Some(*type_id),
        (FiberTag::Fragment, Kind::Fragment(_)) => true,
        // A Suspense boundary is matched by its kind alone: React reuses the
        // boundary fiber for every render of it and diffs the two trees below.
        (FiberTag::Suspense, Kind::Suspense(_)) => true,
        // A provider is matched by the context it publishes. React compares
        // the fiber's type, which *is* the context object; Rea keeps the
        // context in the payload, so the payload is what has to match.
        (FiberTag::Provider, Kind::Provider(new)) => match &fiber.memoized {
            Payload::Provider(old) => Rc::ptr_eq(&old.context, &new.context),
            _ => false,
        },
        _ => false,
    }
}

/// The payload an element asks the fiber to render.
fn payload_of<H: Host>(element: &Element<H>) -> Payload<H> {
    match element.kind() {
        Kind::Empty => Payload::None,
        Kind::Host(node) => Payload::Host(Rc::clone(node)),
        Kind::Component { render, .. } => Payload::Component(Rc::clone(render)),
        Kind::Fragment(children) => Payload::Fragment(Rc::clone(children)),
        Kind::Suspense(node) => Payload::Suspense(Rc::clone(node)),
        Kind::Provider(node) => Payload::Provider(Rc::clone(node)),
    }
}

/// Starts the work-in-progress fiber of a committed child.
fn reuse<H: Host>(arena: &mut Arena<H>, current: FiberId, element: &Element<H>) -> FiberId {
    let Some(work) = arena.create_work_in_progress(current) else {
        return create(arena, element);
    };
    if let Some(fiber) = arena.get_mut(work) {
        fiber.pending = payload_of(element);
        fiber.key = element.key().cloned();
        fiber.element_type = element.component_type();
    }
    work
}

/// Creates the fiber of a child that has no committed counterpart.
fn create<H: Host>(arena: &mut Arena<H>, element: &Element<H>) -> FiberId {
    let tag = match element.kind() {
        Kind::Host(_) => FiberTag::Host,
        Kind::Component { .. } => FiberTag::Component,
        Kind::Suspense(_) => FiberTag::Suspense,
        Kind::Provider(_) => FiberTag::Provider,
        // Empty is skipped by the caller; a fragment is the closest node.
        Kind::Empty | Kind::Fragment(_) => FiberTag::Fragment,
    };
    let mut fiber = Fiber::new(tag);
    fiber.key = element.key().cloned();
    fiber.element_type = element.component_type();
    fiber.pending = payload_of(element);
    fiber.memoized = fiber.pending.clone();
    fiber.flags.insert(Flags::PLACEMENT);
    arena.alloc(fiber)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{AnyContext, Context};

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct TestNode(u32);

    /// Host elements are `(type, props)`: two elements of the same type match.
    struct TestHost;

    impl Host for TestHost {
        type Element = (u8, u16);
        type Node = TestNode;

        fn element_matches(&self, old: &(u8, u16), new: &(u8, u16)) -> bool {
            old.0 == new.0
        }

        fn create(&mut self, _element: &(u8, u16)) -> TestNode {
            TestNode(0)
        }

        fn update(&mut self, _node: TestNode, _element: &(u8, u16)) {}

        fn insert_before(&mut self, _parent: TestNode, _child: TestNode, _anchor: Option<TestNode>) {}

        fn remove(&mut self, _parent: TestNode, _child: TestNode) {}
    }

    type TestElement = Element<TestHost>;

    fn mount(children: &[TestElement]) -> (Arena<TestHost>, FiberId) {
        let mut arena = Arena::new();
        let parent = arena.alloc(Fiber::new(FiberTag::Host));
        reconcile_children(&TestHost, &mut arena, parent, None, children);
        (arena, parent)
    }

    /// Renders `children` again, as the work loop will once it exists.
    fn render(arena: &mut Arena<TestHost>, parent: FiberId, children: &[TestElement]) -> Option<FiberId> {
        let current_first = arena
            .get(parent)
            .and_then(|fiber| fiber.first_child);
        let work = arena.create_work_in_progress(parent)?;
        reconcile_children(&TestHost, arena, work, current_first, children)
    }

    fn chain(arena: &Arena<TestHost>, first: Option<FiberId>) -> alloc::vec::Vec<FiberId> {
        let mut ids = alloc::vec::Vec::new();
        let mut next = first;
        while let Some(id) = next {
            ids.push(id);
            next = arena.get(id).and_then(|fiber| fiber.sibling);
        }
        ids
    }

    fn children(arena: &Arena<TestHost>, parent: FiberId) -> alloc::vec::Vec<FiberId> {
        chain(
            arena,
            arena
                .get(parent)
                .and_then(|fiber| fiber.first_child),
        )
    }

    fn leaf(props: u16) -> TestElement {
        Element::host((0, props))
    }

    fn keyed(props: u16, key: &str) -> TestElement {
        Element::host((0, props)).keyed(key)
    }

    fn panel() -> TestElement {
        Element::component(|_| Element::empty())
    }

    #[test]
    fn mounting_links_the_children_in_order() {
        let (arena, parent) = mount(&[leaf(1), leaf(2), leaf(3)]);
        let ids = children(&arena, parent);
        assert_eq!(ids.len(), 3);
        for (index, id) in ids.iter().enumerate() {
            let Some(fiber) = arena.get(*id) else {
                return;
            };
            assert_eq!(fiber.index, index);
            assert_eq!(fiber.parent, Some(parent));
            assert!(fiber.flags.contains(Flags::PLACEMENT));
        }
        assert!(
            arena
                .get(parent)
                .is_some_and(|fiber| fiber.deletions.is_empty())
        );
    }

    #[test]
    fn an_empty_element_takes_no_slot() {
        let (arena, parent) = mount(&[leaf(1), Element::empty(), leaf(2)]);
        let ids = children(&arena, parent);
        assert_eq!(ids.len(), 2);
        assert_eq!(
            ids.last()
                .and_then(|id| arena.get(*id))
                .map(|fiber| fiber.index),
            Some(1)
        );
    }

    #[test]
    fn a_prefix_is_reused() {
        let (mut arena, parent) = mount(&[leaf(1), leaf(2)]);
        let old = children(&arena, parent);
        let first = render(&mut arena, parent, &[leaf(1), leaf(2)]);
        let new = chain(&arena, first);
        assert_eq!(new.len(), 2);
        for (new_id, old_id) in new.iter().zip(&old) {
            let Some(fiber) = arena.get(*new_id) else {
                return;
            };
            assert_eq!(fiber.alternate, Some(*old_id));
            assert!(!fiber.flags.contains(Flags::PLACEMENT));
        }
    }

    #[test]
    fn a_child_keeps_its_fiber_when_its_props_change() {
        let (mut arena, parent) = mount(&[leaf(1)]);
        let old = children(&arena, parent);
        let first = render(&mut arena, parent, &[leaf(2)]);
        let new = chain(&arena, first);
        assert_eq!(new.len(), 1);
        let Some(fiber) = arena.get(new[0]) else {
            return;
        };
        assert_eq!(fiber.alternate, Some(old[0]));
        assert!(matches!(&fiber.pending, Payload::Host(node) if node.element.1 == 2));
        assert!(matches!(&fiber.memoized, Payload::Host(node) if node.element.1 == 1));
    }

    #[test]
    fn a_reorder_keeps_identity_and_places_only_the_child_that_moved_back() {
        let (mut arena, parent) = mount(&[keyed(1, "a"), keyed(1, "b"), keyed(1, "c")]);
        let old = children(&arena, parent);
        let first = render(&mut arena, parent, &[keyed(1, "b"), keyed(1, "c"), keyed(1, "a")]);
        let new = chain(&arena, first);
        assert_eq!(new.len(), 3);
        let Some(b) = arena.get(new[0]) else {
            return;
        };
        let Some(c) = arena.get(new[1]) else {
            return;
        };
        let Some(a) = arena.get(new[2]) else {
            return;
        };
        assert_eq!(b.alternate, Some(old[1]));
        assert_eq!(c.alternate, Some(old[2]));
        assert_eq!(a.alternate, Some(old[0]));
        assert!(!b.flags.contains(Flags::PLACEMENT));
        assert!(!c.flags.contains(Flags::PLACEMENT));
        assert!(a.flags.contains(Flags::PLACEMENT));
        assert!(
            arena
                .get(parent)
                .is_some_and(|fiber| fiber.deletions.is_empty())
        );
    }

    #[test]
    fn an_insertion_creates_only_the_new_child() {
        let (mut arena, parent) = mount(&[keyed(1, "a"), keyed(1, "b")]);
        let old = children(&arena, parent);
        let first = render(&mut arena, parent, &[keyed(1, "a"), keyed(1, "x"), keyed(1, "b")]);
        let new = chain(&arena, first);
        assert_eq!(new.len(), 3);
        let Some(a) = arena.get(new[0]) else {
            return;
        };
        let Some(x) = arena.get(new[1]) else {
            return;
        };
        let Some(b) = arena.get(new[2]) else {
            return;
        };
        assert_eq!(a.alternate, Some(old[0]));
        assert_eq!(x.alternate, None);
        assert!(x.flags.contains(Flags::PLACEMENT));
        assert_eq!(b.alternate, Some(old[1]));
    }

    #[test]
    fn a_removed_child_is_reported_to_the_parent() {
        let (mut arena, parent) = mount(&[keyed(1, "a"), keyed(1, "b"), keyed(1, "c")]);
        let old = children(&arena, parent);
        let first = render(&mut arena, parent, &[keyed(1, "a"), keyed(1, "c")]);
        let new = chain(&arena, first);
        assert_eq!(new.len(), 2);
        let Some(work) = arena
            .get(parent)
            .and_then(|fiber| fiber.alternate)
        else {
            return;
        };
        let Some(fiber) = arena.get(work) else {
            return;
        };
        assert_eq!(fiber.deletions, [old[1]]);
        assert!(fiber.flags.contains(Flags::CHILD_DELETION));
    }

    #[test]
    fn a_changed_host_type_replaces_the_child() {
        let (mut arena, parent) = mount(&[Element::host((0, 1))]);
        let old = children(&arena, parent);
        let first = render(&mut arena, parent, &[Element::host((1, 1))]);
        let new = chain(&arena, first);
        assert_eq!(new.len(), 1);
        let Some(fiber) = arena.get(new[0]) else {
            return;
        };
        assert_eq!(fiber.alternate, None);
        assert!(fiber.flags.contains(Flags::PLACEMENT));
        let Some(work) = arena
            .get(parent)
            .and_then(|fiber| fiber.alternate)
        else {
            return;
        };
        assert!(
            arena
                .get(work)
                .is_some_and(|fiber| fiber.deletions == [old[0]])
        );
    }

    #[test]
    fn a_component_is_matched_by_its_function_type() {
        let (mut arena, parent) = mount(&[panel()]);
        let old = children(&arena, parent);
        let first = render(&mut arena, parent, &[panel()]);
        let same = chain(&arena, first);
        assert_eq!(same.len(), 1);
        assert_eq!(
            arena
                .get(same[0])
                .and_then(|fiber| fiber.alternate),
            Some(old[0])
        );

        let other = Element::component(|_| leaf(9));
        let first = render(&mut arena, parent, &[other]);
        let replaced = chain(&arena, first);
        assert_eq!(replaced.len(), 1);
        assert_eq!(
            arena
                .get(replaced[0])
                .and_then(|fiber| fiber.alternate),
            None
        );
    }

    #[test]
    fn a_suspense_element_gets_a_suspense_fiber() {
        let (arena, parent) = mount(&[Element::suspense(leaf(1), leaf(2))]);
        let ids = children(&arena, parent);
        assert_eq!(ids.len(), 1);
        let Some(fiber) = arena.get(ids[0]) else {
            return;
        };
        assert_eq!(fiber.tag, FiberTag::Suspense);
        assert!(matches!(&fiber.pending, Payload::Suspense(node) if node.children.key().is_none()));
    }

    #[test]
    fn a_suspense_boundary_is_reused_by_its_kind() {
        let (mut arena, parent) = mount(&[Element::suspense(leaf(1), leaf(2))]);
        let old = children(&arena, parent);
        let first = render(&mut arena, parent, &[Element::suspense(leaf(3), leaf(4))]);
        let new = chain(&arena, first);
        assert_eq!(new.len(), 1);
        assert_eq!(
            arena
                .get(new[0])
                .and_then(|fiber| fiber.alternate),
            Some(old[0])
        );
    }

    #[test]
    fn a_fragment_carries_its_children_into_its_fiber() {
        let (arena, parent) = mount(&[Element::fragment([leaf(1), leaf(2)])]);
        let ids = children(&arena, parent);
        assert_eq!(ids.len(), 1);
        let Some(fiber) = arena.get(ids[0]) else {
            return;
        };
        assert_eq!(fiber.tag, FiberTag::Fragment);
        assert!(matches!(&fiber.pending, Payload::Fragment(children) if children.len() == 2));
    }

    fn container(nested: alloc::vec::Vec<TestElement>) -> TestElement {
        Element::host((0, 0)).children(nested)
    }

    #[test]
    fn a_host_element_carries_the_children_it_arranges() {
        let (arena, parent) = mount(&[container(alloc::vec![leaf(1), leaf(2)])]);
        let hosts = children(&arena, parent);
        assert_eq!(hosts.len(), 1);
        let Some(fiber) = arena.get(hosts[0]) else {
            return;
        };
        assert_eq!(fiber.tag, FiberTag::Host);
        // The work pass diffs these against the host's committed children when
        // it begins the host, so the payload has to keep them.
        assert!(matches!(&fiber.pending, Payload::Host(node) if node.children.len() == 2));
        assert!(children(&arena, hosts[0]).is_empty());
    }

    #[test]
    fn the_committed_tree_is_left_alone() {
        let (mut arena, parent) = mount(&[keyed(1, "a"), keyed(1, "b"), keyed(1, "c")]);
        let old = children(&arena, parent);
        render(&mut arena, parent, &[keyed(1, "c"), keyed(1, "b"), keyed(1, "a")]);
        let Some(a) = arena.get(old[0]) else {
            return;
        };
        assert_eq!(a.sibling, Some(old[1]));
        assert_eq!(a.parent, Some(parent));
        assert_eq!(a.index, 0);
        assert_eq!(
            arena
                .get(parent)
                .and_then(|fiber| fiber.first_child),
            Some(old[0])
        );
        assert!(
            arena
                .get(parent)
                .is_some_and(|fiber| fiber.deletions.is_empty())
        );
    }

    #[test]
    fn a_provider_element_gets_a_provider_fiber() {
        let context = Context::new(0_u16);
        let (arena, parent) = mount(&[Element::provider(&context, Rc::new(1), [leaf(1)])]);
        let ids = children(&arena, parent);
        assert_eq!(ids.len(), 1);
        let Some(fiber) = arena.get(ids[0]) else {
            return;
        };
        assert_eq!(fiber.tag, FiberTag::Provider);
        assert!(fiber.flags.contains(Flags::PLACEMENT));
        // The payload carries the context, because the fiber's begin has to
        // know which cell to publish the value into.
        let Payload::Provider(node) = &fiber.pending else {
            return;
        };
        assert!(Rc::ptr_eq(
            &node.context,
            &(Rc::clone(&context) as Rc<dyn AnyContext>)
        ));
        assert_eq!(node.children.len(), 1);
    }

    #[test]
    fn a_provider_is_reused_by_the_context_it_publishes() {
        let context = Context::new(0_u16);
        let (mut arena, parent) = mount(&[Element::provider(&context, Rc::new(1), [leaf(1)])]);
        let old = children(&arena, parent);
        let first = render(
            &mut arena,
            parent,
            &[Element::provider(&context, Rc::new(2), [leaf(2)])],
        );
        let new = chain(&arena, first);
        assert_eq!(new.len(), 1);
        assert_eq!(
            arena
                .get(new[0])
                .and_then(|fiber| fiber.alternate),
            Some(old[0])
        );
        let Some(fiber) = arena.get(new[0]) else {
            return;
        };
        assert!(matches!(&fiber.pending, Payload::Provider(node) if node.children.len() == 1));
    }

    #[test]
    fn a_provider_of_another_context_takes_a_new_fiber() {
        let first_context = Context::new(0_u16);
        let second_context = Context::new(0_u16);
        let (mut arena, parent) = mount(&[Element::provider(&first_context, Rc::new(1), [leaf(1)])]);
        let old = children(&arena, parent);
        let first = render(
            &mut arena,
            parent,
            &[Element::provider(&second_context, Rc::new(1), [leaf(1)])],
        );
        let new = chain(&arena, first);
        assert_eq!(new.len(), 1);
        // Two contexts with the same default are still two contexts: a provider
        // of one cannot publish the other's value, so the fiber is replaced the
        // way React replaces a provider whose type changed.
        assert_eq!(
            arena
                .get(new[0])
                .and_then(|fiber| fiber.alternate),
            None
        );
        assert_ne!(new[0], old[0]);
    }
}
