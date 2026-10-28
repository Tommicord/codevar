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

//! The element tree.
//!
//! An element is an immutable description of one node of the interface. A
//! component returns elements; the reconciler turns them into fibers. The
//! description is cheap to build and cheap to clone — a host element and a
//! fragment's children sit behind an [`Rc`] — so a render can produce a whole
//! tree without copying the previous one.
//!
//! # Identity
//!
//! Three things decide whether an element can reuse the fiber it met last
//! time: its [`Key`], its position, and its type. The type is the concrete
//! element type for a host node, and [`TypeId`] for a component. A component
//! written as a named function and a component written as a closure both have
//! a stable type as long as the same source expression creates them, which is
//! what makes `Element::component(|ctx| ...)` inside a render stable across
//! renders.
//!
//! # Empty
//!
//! A component that renders nothing returns [`Element::empty`]. The
//! reconciler skips empty elements entirely: they never get a fiber, and they
//! do not shift the position of their siblings.
//!
//! # No widgets here
//!
//! The engine knows structure and nothing else. It never names a widget, a
//! label, a button or a renderer: a leaf is whatever [`Host::Element`] is, and
//! only the host that defines it can say how two leaves match, how one is
//! created and how it is updated. Even a text run is a host element, because
//! the host's text widget carries the font, the color and the layout that the
//! engine must not see. Adding a widget therefore never changes this crate.
//!
//! A host element may also carry the elements it arranges — the children of a
//! column, a row or a frame. The children belong to the description, not to a
//! component: the reconciler diffs them against the children of the host fiber
//! when it reaches the host, exactly as it diffs the children of a fragment,
//! so a host node keeps the fibers and nodes below it across a re-render.

use alloc::rc::Rc;
use core::any::TypeId;

use crate::hooks::{Cleanup, RenderCtx};
use crate::host::{Host, Key};

/// The render function of a component.
///
/// The function is higher-ranked over both of the render scope's lifetimes:
/// the scope is only valid for one call, and a component cannot keep the
/// borrow it is given.
pub type ComponentFn<H> = dyn Fn(&mut RenderCtx<'_>) -> Element<H>;

/// The callback a host element hands its node to.
///
/// The callback receives the node while the element is mounted and `None` when
/// the node is detached. It may return a cleanup, which then runs on detach in
/// place of the `None` call, exactly like a React 19 callback ref.
///
/// The callback is behind an `Rc`, so its identity — not the source expression
/// that created it — is what decides whether a render changed it. A component
/// must pass the same `Rc` on every render, just as a callback hoisted out of a
/// React component is the same function object; a fresh `Rc` per render is a
/// different callback and is detached and attached again.
pub type Reference<H> = Rc<dyn Fn(Option<<H as Host>::Node>) -> Option<Cleanup>>;

/// One node of the tree a component describes.
pub struct Element<H: Host> {
    /// The identity of the element among its siblings, when it has one.
    key: Option<Key>,
    /// What the element describes.
    kind: Kind<H>,
}

/// A host element together with the element children it arranges.
///
/// The description of the node itself is what the host matches and updates;
/// the children are reconciled into the host's fiber, so the host node keeps
/// them from one render to the next.
pub(crate) struct HostNode<H: Host> {
    /// The host's description of the node.
    pub(crate) element: Rc<H::Element>,
    /// The elements the node arranges.
    pub(crate) children: Rc<[Element<H>]>,
    /// The callback the node is handed to, when the element has one.
    pub(crate) reference: Option<Reference<H>>,
}

/// What an element describes.
pub(crate) enum Kind<H: Host> {
    /// Renders nothing.
    Empty,
    /// An opaque host element and the children it arranges.
    Host(Rc<HostNode<H>>),
    /// A component, with the identity that decides whether a fiber can be
    /// reused when the parent renders again.
    Component {
        /// The function that renders the component.
        render: Rc<ComponentFn<H>>,
        /// The type of the function, used as the component's identity.
        type_id: TypeId,
    },
    /// A group of children with no node of its own.
    Fragment(Rc<[Element<H>]>),
}

impl<H: Host> Element<H> {
    /// An element that renders nothing.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            key: None,
            kind: Kind::Empty,
        }
    }

    /// An opaque host element.
    ///
    /// The engine never looks inside it. The host decides what can appear
    /// here: a widget, a text run, an image, or a whole subtree it renders
    /// itself.
    #[must_use]
    pub fn host(element: H::Element) -> Self {
        Self {
            key: None,
            kind: Kind::Host(Rc::new(HostNode {
                element: Rc::new(element),
                children: Rc::from([]),
                reference: None,
            })),
        }
    }

    /// Gives a host element the children it arranges.
    ///
    /// The children are elements of their own, reconciled into the host's
    /// fiber when the host is reached. Only a host element arranges children
    /// this way: a component describes its children by rendering them, and a
    /// fragment takes its children in [`Element::fragment`], so the call is
    /// ignored on any other kind.
    #[must_use]
    pub fn children(mut self, children: impl IntoIterator<Item = Self>) -> Self {
        if let Kind::Host(node) = &self.kind {
            self.kind = Kind::Host(Rc::new(HostNode {
                element: Rc::clone(&node.element),
                children: children.into_iter().collect(),
                reference: node.reference.clone(),
            }));
        }
        self
    }

    /// Gives a host element a reference callback.
    ///
    /// The callback is attached in the layout phase and detached in the
    /// mutation phase; see [`Reference`] for what it receives. As with
    /// [`Element::children`], the call is ignored on any other kind, because
    /// only a host element owns a node to reference.
    #[must_use]
    pub fn referenced(mut self, reference: Reference<H>) -> Self {
        if let Kind::Host(node) = &self.kind {
            self.kind = Kind::Host(Rc::new(HostNode {
                element: Rc::clone(&node.element),
                children: Rc::clone(&node.children),
                reference: Some(reference),
            }));
        }
        self
    }

    /// A component.
    ///
    /// The closure must be the same source expression on every render of the
    /// parent; a closure created inside a loop with different captures is a
    /// different component to the reconciler, exactly as an inline component
    /// function is in React.
    #[must_use]
    pub fn component<F>(component: F) -> Self
    where
        F: Fn(&mut RenderCtx<'_>) -> Element<H> + 'static,
    {
        let type_id = TypeId::of::<F>();
        Self {
            key: None,
            kind: Kind::Component {
                render: Rc::new(component),
                type_id,
            },
        }
    }

    /// A group of children with no node of its own.
    #[must_use]
    pub fn fragment(children: impl IntoIterator<Item = Self>) -> Self {
        Self {
            key: None,
            kind: Kind::Fragment(children.into_iter().collect()),
        }
    }

    /// Gives the element a key, so a reorder can follow it.
    #[must_use]
    pub fn keyed(mut self, key: impl Into<Key>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// The key of the element, when it has one.
    #[must_use]
    pub fn key(&self) -> Option<&Key> {
        self.key.as_ref()
    }

    /// Whether the element renders nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self.kind, Kind::Empty)
    }

    /// The component identity, when the element is a component.
    #[must_use]
    pub(crate) fn component_type(&self) -> Option<TypeId> {
        match &self.kind {
            Kind::Component { type_id, .. } => Some(*type_id),
            _ => None,
        }
    }

    /// What the element describes.
    pub(crate) const fn kind(&self) -> &Kind<H> {
        &self.kind
    }
}

impl<H: Host> Clone for Element<H> {
    fn clone(&self) -> Self {
        let kind = match &self.kind {
            Kind::Empty => Kind::Empty,
            Kind::Host(node) => Kind::Host(Rc::clone(node)),
            Kind::Component { render, type_id } => Kind::Component {
                render: Rc::clone(render),
                type_id: *type_id,
            },
            Kind::Fragment(children) => Kind::Fragment(Rc::clone(children)),
        };
        Self {
            key: self.key.clone(),
            kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct TestNode(u32);

    struct TestHost;

    impl Host for TestHost {
        type Element = u32;
        type Node = TestNode;

        fn element_matches(&self, old: &u32, new: &u32) -> bool {
            old == new
        }

        fn create(&mut self, _element: &u32) -> TestNode {
            TestNode(0)
        }

        fn update(&mut self, _node: TestNode, _element: &u32) {}

        fn insert_before(&mut self, _parent: TestNode, _child: TestNode, _anchor: Option<TestNode>) {}

        fn remove(&mut self, _parent: TestNode, _child: TestNode) {}
    }

    fn panel() -> Element<TestHost> {
        Element::component(|_| Element::empty())
    }

    #[test]
    fn an_unkeyed_element_has_no_key() {
        let element: Element<TestHost> = Element::host(3);
        assert_eq!(element.key(), None);
        assert!(!element.is_empty());
    }

    #[test]
    fn a_key_follows_the_element() {
        let element: Element<TestHost> = Element::host(1).keyed("row-1");
        assert_eq!(element.key().map(Key::as_str), Some("row-1"));
    }

    #[test]
    fn an_empty_element_renders_nothing() {
        let element: Element<TestHost> = Element::empty();
        assert!(element.is_empty());
    }

    #[test]
    fn cloning_shares_the_host_description() {
        let element: Element<TestHost> = Element::host(7);
        let clone = element.clone();
        let (Kind::Host(a), Kind::Host(b)) = (element.kind(), clone.kind()) else {
            return;
        };
        assert!(Rc::ptr_eq(a, b));
        assert_eq!(*a.element, 7);
    }

    #[test]
    fn a_host_element_keeps_the_children_it_is_given() {
        let element: Element<TestHost> =
            Element::host(1).children([Element::host(2), Element::host(3).keyed("b")]);
        let Kind::Host(node) = element.kind() else {
            return;
        };
        assert_eq!(node.children.len(), 2);
        assert!(matches!(node.children[0].kind(), Kind::Host(child) if *child.element == 2));
        assert_eq!(node.children[1].key().map(Key::as_str), Some("b"));
    }

    #[test]
    fn adding_children_leaves_the_original_element_alone() {
        let original: Element<TestHost> = Element::host(4);
        let parent = original.clone().children([Element::host(5)]);
        let (Kind::Host(a), Kind::Host(b)) = (original.kind(), parent.kind()) else {
            return;
        };
        assert!(a.children.is_empty());
        assert_eq!(b.children.len(), 1);
        assert!(Rc::ptr_eq(&a.element, &b.element));
    }

    #[test]
    fn a_reference_follows_the_children_it_is_given() {
        let reference: Reference<TestHost> = Rc::new(|_| None);
        let element: Element<TestHost> = Element::host(1)
            .referenced(Rc::clone(&reference))
            .children([Element::host(2)]);
        let Kind::Host(node) = element.kind() else {
            return;
        };
        assert!(
            node.reference
                .as_ref()
                .is_some_and(|kept| Rc::ptr_eq(kept, &reference))
        );
        assert_eq!(node.children.len(), 1);
    }

    #[test]
    fn a_component_ignores_a_reference() {
        let reference: Reference<TestHost> = Rc::new(|_| None);
        let element: Element<TestHost> = panel().referenced(reference);
        assert!(element.component_type().is_some());
    }

    #[test]
    fn a_component_ignores_children_it_cannot_arrange() {
        let element: Element<TestHost> = panel().children([Element::host(1)]);
        assert!(element.component_type().is_some());
    }

    #[test]
    fn the_same_component_expression_keeps_its_identity() {
        let first = panel();
        let second = panel();
        assert_eq!(first.component_type(), second.component_type());
        assert!(first.component_type().is_some());
    }

    #[test]
    fn different_component_expressions_differ() {
        let first = panel();
        let second: Element<TestHost> = Element::component(|_| Element::host(9));
        assert_ne!(first.component_type(), second.component_type());
    }

    #[test]
    fn a_fragment_keeps_its_children_in_order() {
        let fragment: Element<TestHost> = Element::fragment([Element::host(1), Element::host(2)]);
        let Kind::Fragment(children) = fragment.kind() else {
            return;
        };
        assert_eq!(children.len(), 2);
        assert!(matches!(children[0].kind(), Kind::Host(element) if *element.element == 1));
        assert!(matches!(children[1].kind(), Kind::Host(element) if *element.element == 2));
    }
}
