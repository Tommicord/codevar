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

//! The host boundary.
//!
//! Rea reconciles a declarative tree without knowing what the leaves mean.
//! [`Host`] is the contract that turns a host element into a node of the real
//! tree and applies the mutations the commit phase asks for. `codevar-kuitui`
//! implements it on top of its retained widget tree; a test harness implements
//! it on top of a vector.
//!
//! The boundary is deliberately narrow. It has no query methods: the
//! reconciler keeps the structure it needs itself, so the host is never asked
//! to walk its own children. That is what makes the host cheap to implement
//! and keeps the number of host calls proportional to the mutations, not to the
//! size of the tree.
//!
//! # Ownership
//!
//! The host owns its nodes. Rea only stores the [`Host::Node`] handle in the
//! fiber and passes it back verbatim. A node is created by [`Host::create`] and
//! lives until the commit phase removes it with [`Host::remove`]; rea never
//! drops a node on its own.
//!
//! # Children
//!
//! [`Host::insert_before`] is the only way to attach a node. It must move the
//! node when it is already attached somewhere, so the commit phase can reorder
//! children without first removing them. Passing `None` as the anchor appends.

use alloc::rc::Rc;
use alloc::string::{String, ToString};
use core::any::Any;
use core::fmt;

/// The props of a component, type-erased.
///
/// A component declares the type of its props and the element builder stores
/// them in a `Rc<dyn Any>`. The render function downcasts them back to the
/// concrete type; a mismatch is a programming error in the component's own
/// builder, not something the engine can recover from.
pub type Props = Rc<dyn Any>;

/// Erases the props of a component.
#[must_use]
pub fn props<T: 'static>(value: T) -> Props {
    Rc::new(value)
}

/// Borrows the props of a component as their concrete type.
#[must_use]
pub fn props_ref<T: 'static>(props: &Props) -> Option<&T> {
    props.downcast_ref::<T>()
}

/// The identity of a child inside its parent.
///
/// Keys let the reconciler follow a child across a reorder, so a list can move
/// an item instead of destroying and recreating it. Two children of the same
/// parent must not share a key; an unkeyed child is matched by its position.
///
/// A key is compared by value and cloned cheaply.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Key(Rc<str>);

impl Key {
    /// Creates a key from a string.
    #[must_use]
    pub fn new(key: impl Into<String>) -> Self {
        Self(Rc::from(key.into()))
    }

    /// The key as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Key {
    fn from(key: &str) -> Self {
        Self(Rc::from(key))
    }
}

impl From<String> for Key {
    fn from(key: String) -> Self {
        Self(Rc::from(key))
    }
}

impl From<u64> for Key {
    fn from(key: u64) -> Self {
        Self(Rc::from(key.to_string()))
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Key")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for Key {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// The tree the reconciler commits into.
///
/// The associated types keep the engine free of a concrete node
/// representation. `Element` is the host's description of a node — a widget
/// and its modifiers, a paragraph, an image — and `Node` is the handle the
/// host uses to address what it created.
///
/// # Hosts and widgets
///
/// The trait says nothing about widgets. It has no method for a label, a
/// button, an image or a text run, and this crate never names one: everything
/// a host shows travels through [`Element`](Host::Element), which the engine
/// treats as opaque. Adding a widget to a host never changes this crate.
///
/// # Implementor contract
///
/// * [`create`](Host::create) returns a node with no children. The commit
///   phase attaches the children itself, in order, with
///   [`insert_before`](Host::insert_before).
/// * [`update`](Host::update) changes only the node itself. It is allowed to
///   replace the node's content, but not to reorder or remove its children.
/// * [`element_matches`](Host::element_matches) decides whether an existing
///   node of type `old` can be reused for `new`. It must compare the *kind* of
///   the element, never its props: two text labels with different strings are
///   the same kind, while a label and an image are not.
/// * [`insert_before`](Host::insert_before) moves a node that is already
///   attached. Passing `None` as `anchor` appends.
/// * A node may have at most one parent. The commit phase never asks for a node
///   to be inserted in two places.
pub trait Host {
    /// The host's description of a node.
    ///
    /// The reconciler stores it in an `Rc` and passes it by reference, so it
    /// does not have to be cheap to clone.
    type Element: 'static;

    /// A handle to a node owned by the host.
    ///
    /// A handle is copied into fibers, so it must be `Copy`. It must identify
    /// the node uniquely for as long as the node is alive.
    type Node: Copy + PartialEq + Eq;

    /// Whether an existing host node of kind `old` can be reused for `new`.
    ///
    /// Compare the kind of the element, not its props.
    fn element_matches(&self, old: &Self::Element, new: &Self::Element) -> bool;

    /// Creates a node for `element`, without children.
    fn create(&mut self, element: &Self::Element) -> Self::Node;

    /// Replaces the description of `node` with `element`.
    ///
    /// The node keeps its children; the commit phase reconciles them itself.
    fn update(&mut self, node: Self::Node, element: &Self::Element);

    /// Inserts `child` under `parent` before `anchor`, or appends it.
    ///
    /// `child` is detached by the caller; when it is still attached, the host
    /// must move it. `anchor` is either `None` or a direct child of `parent`.
    fn insert_before(&mut self, parent: Self::Node, child: Self::Node, anchor: Option<Self::Node>);

    /// Removes `child` from `parent`.
    ///
    /// The host may reclaim the node now; rea drops its handle after this call.
    fn remove(&mut self, parent: Self::Node, child: Self::Node);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_compares_by_value() {
        assert_eq!(Key::from("item"), Key::new("item"));
        assert_ne!(Key::from("item"), Key::from("other"));
        assert_eq!(Key::from(7_u64).as_str(), "7");
        assert_eq!(Key::from("a").to_string(), "a");
    }

    #[test]
    fn props_round_trip_through_their_type() {
        let props = props(41_u32);
        assert_eq!(props_ref::<u32>(&props), Some(&41));
        assert!(props_ref::<u8>(&props).is_none());
    }
}
