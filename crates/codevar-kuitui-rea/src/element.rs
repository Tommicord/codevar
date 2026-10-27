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

// The reconciler that reads these descriptions lands in the next increment.
#![allow(dead_code)]

use alloc::rc::Rc;
use alloc::string::String;
use core::any::TypeId;

use crate::hooks::RenderCtx;
use crate::host::{Host, Key};

/// The render function of a component.
///
/// The function is higher-ranked over both of the render scope's lifetimes:
/// the scope is only valid for one call, and a component cannot keep the
/// borrow it is given.
pub type ComponentFn<H> = dyn Fn(&mut RenderCtx<'_>) -> Element<H>;

/// One node of the tree a component describes.
pub struct Element<H: Host> {
    /// The identity of the element among its siblings, when it has one.
    key: Option<Key>,
    /// What the element describes.
    kind: Kind<H>,
}

/// What an element describes.
pub(crate) enum Kind<H: Host> {
    /// Renders nothing.
    Empty,
    /// A text run.
    Text(Rc<str>),
    /// A host element.
    Host(Rc<H::Element>),
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

    /// A text run.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            key: None,
            kind: Kind::Text(Rc::from(text.into())),
        }
    }

    /// A host element.
    #[must_use]
    pub fn host(element: H::Element) -> Self {
        Self {
            key: None,
            kind: Kind::Host(Rc::new(element)),
        }
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
            Kind::Text(text) => Kind::Text(Rc::clone(text)),
            Kind::Host(element) => Kind::Host(Rc::clone(element)),
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

        fn create_text(&mut self, _text: &str) -> TestNode {
            TestNode(0)
        }

        fn update_text(&mut self, _node: TestNode, _text: &str) {}

        fn insert_before(&mut self, _parent: TestNode, _child: TestNode, _anchor: Option<TestNode>) {}

        fn remove(&mut self, _parent: TestNode, _child: TestNode) {}
    }

    fn panel() -> Element<TestHost> {
        Element::component(|_| Element::empty())
    }

    #[test]
    fn an_unkeyed_element_has_no_key() {
        let element: Element<TestHost> = Element::text("hello");
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
        let second: Element<TestHost> = Element::component(|_| Element::text("other"));
        assert_ne!(first.component_type(), second.component_type());
    }

    #[test]
    fn a_fragment_keeps_its_children_in_order() {
        let fragment: Element<TestHost> = Element::fragment([Element::text("a"), Element::host(2)]);
        let Kind::Fragment(children) = fragment.kind() else {
            return;
        };
        assert_eq!(children.len(), 2);
        assert!(matches!(children[0].kind(), Kind::Text(text) if &**text == "a"));
        assert!(matches!(children[1].kind(), Kind::Host(element) if **element == 2));
    }
}
