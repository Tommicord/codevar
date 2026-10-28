//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The widget host: the boundary the reactivity engine commits into.
//!
//! `codevar-kuitui-rea` reconciles a declarative tree without knowing what a
//! node is. [`WidgetHost`] implements its `Host` trait for the retained widget
//! tree: the host owns one [`Node`] per host element, keeps the child order the
//! engine asked for, and hands the assembled tree to the layout, paint and
//! event passes.
//!
//! # Elements
//!
//! A host element is a [`NodeElement`]: the kind of widget the node holds and a
//! factory that builds it. The kind decides reuse — two labels with different
//! text are the same kind, a label and a button are not — and the factory runs
//! on every create and update, because a widget is described by its value on
//! each render. What survives a render is the node, its position and the hooks
//! above it; a widget that needs to survive has to keep its state in a hook or
//! in a shared value its description carries.
//!
//! # Opaque subtrees
//!
//! Some widget libraries build their nodes outside the engine: a button
//! answers a trait and returns the node that draws it, children included.
//! [`NodeElement::opaque`] describes such a node. The engine reconciles the
//! node and the slot it lives in, but the subtree the factory builds stays the
//! node's own: assembly leaves it where the factory put it, and the element
//! takes no element children. The factory runs on every create and update, so
//! the description the render produced is the one the node is rebuilt from.
//!
//! # Assembly
//!
//! A [`Node`] owns its children, so one node cannot be both in the host's
//! arena and in its parent's child list. The host therefore stores every node
//! *childless* and keeps the structure — the parent and the child order —
//! beside it. [`WidgetHost::assemble`] nests the nodes into the tree a layout
//! pass walks, and [`WidgetHost::disassemble`] takes the tree apart again
//! before the engine mutates anything. An opaque node is the exception: it
//! keeps the children its factory built, and only the nodes above it are
//! nested.
//!
//! The slots keep their ids across the hand-off, so the handle a fiber stores
//! stays valid and a widget keeps its state from one frame to the next. The
//! hand-off itself is a walk over the tree, not a rebuild: no widget is
//! constructed, no child list is reallocated beyond the moves a [`Node`]
//! already owns.
//!
//! # The container
//!
//! The root is a container node holding a [`Stack`], which lays every element
//! the root renders over the same rectangle. With the tight constraints of the
//! window, the top-level element fills it, which is what an application
//! expects; a root that renders a fragment shows its pieces stacked.

use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use core::any::TypeId;
use core::mem;

use codevar_kuitui_rea::Host;

use crate::node::{Modifier, Node};
use crate::widget::Widget;
use crate::widgets::Stack;

/// The handle of a node owned by a [`WidgetHost`].
///
/// The id stays valid for as long as the node is alive; it survives the
/// assemble/disassemble hand-off, so a fiber can keep it between frames.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(u32);

impl NodeId {
    /// The index of the slot the id names.
    const fn index(self) -> usize {
        self.0 as usize
    }
}

/// The description of one widget node.
///
/// The engine compares the widget type to decide whether a committed node can
/// be reused, and calls the factory to build the widget on every create and
/// update. See the [module docs](self) for why the widget is rebuilt instead
/// of patched.
#[derive(Clone)]
pub struct NodeElement {
    /// The widget type, which is the identity the engine compares.
    kind: TypeId,
    /// The factory of the widget.
    make: Rc<dyn Fn() -> Node>,
    /// The modifier chain the built node carries.
    modifier: Modifier,
    /// Whether the node owns the subtree under it.
    opaque: bool,
}

impl NodeElement {
    /// Describes a node holding a `W`.
    ///
    /// `make` builds the widget and runs on every create and update. `W` is
    /// the identity the engine compares: an element of the same `W` reuses the
    /// committed node, a different one replaces it.
    pub fn new<W: Widget + 'static>(make: impl Fn() -> W + 'static) -> Self {
        Self {
            kind: TypeId::of::<W>(),
            make: Rc::new(move || Node::new(make())),
            modifier: Modifier::none(),
            opaque: false,
        }
    }

    /// Describes a node that owns the subtree under it.
    ///
    /// The engine reconciles the node but never the children `make` builds:
    /// assembly leaves that subtree where the factory put it, so a library
    /// that builds nodes of its own — a button that answers a trait, for
    /// example — is placed in a reactive tree as it is. The factory runs on
    /// every create and update, which is what makes the node follow the
    /// description the last render produced.
    ///
    /// `M` is the identity the engine compares: two opaque elements of the
    /// same `M` reuse one slot, and an opaque element never matches a typed
    /// one. An opaque element takes no element children; everything under the
    /// node is the factory's.
    pub fn opaque<M: 'static>(make: impl Fn() -> Node + 'static) -> Self {
        Self {
            kind: TypeId::of::<M>(),
            make: Rc::new(make),
            modifier: Modifier::none(),
            opaque: true,
        }
    }

    /// Decorates the node with a modifier chain.
    ///
    /// The chain is cloneable, so it is applied to the node the factory builds
    /// on every create and update.
    #[must_use]
    pub fn modifier(mut self, modifier: Modifier) -> Self {
        self.modifier = modifier;
        self
    }

    /// Builds the node this element describes.
    fn build(&self) -> Node {
        (self.make)().modifier(self.modifier.clone())
    }
}

/// One slot of the host: a node and the structure that surrounds it.
struct Slot {
    /// The node, or `None` while the assembled tree holds it.
    node: Option<Node>,
    /// The parent the node is attached to, when it is attached.
    parent: Option<NodeId>,
    /// The children, in order.
    children: Vec<NodeId>,
    /// Whether the node owns the subtree under it.
    opaque: bool,
}

/// The host that owns the widget nodes of one tree.
///
/// The host implements `codevar_kuitui_rea::Host`; an application drives it
/// through the composition, which calls [`assemble`](Self::assemble) before a
/// layout pass and [`disassemble`](Self::disassemble) before a render.
pub struct WidgetHost {
    /// The slots, indexed by [`NodeId`].
    slots: Vec<Slot>,
    /// The ids of the slots a removal freed.
    free: Vec<NodeId>,
    /// The node the root's elements are inserted into.
    container: NodeId,
    /// Whether the assembled tree holds the nodes.
    assembled: bool,
}

impl WidgetHost {
    /// Creates a host whose root is an empty stack.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: vec![Slot {
                node: Some(Node::new(Stack::new())),
                parent: None,
                children: Vec::new(),
                opaque: false,
            }],
            free: Vec::new(),
            container: NodeId(0),
            assembled: false,
        }
    }

    /// The node the elements of the root are inserted into.
    #[must_use]
    pub const fn container(&self) -> NodeId {
        self.container
    }

    /// The children of a node, in order.
    #[must_use]
    pub fn children(&self, node: NodeId) -> &[NodeId] {
        self.slots
            .get(node.index())
            .map_or(&[], |slot| &slot.children)
    }

    /// The node a handle names, when the host owns it.
    ///
    /// `None` while the assembled tree holds the nodes.
    #[must_use]
    pub fn node(&self, node: NodeId) -> Option<&Node> {
        self.slots
            .get(node.index())
            .and_then(|slot| slot.node.as_ref())
    }

    /// The node a handle names, for mutation.
    #[must_use]
    pub fn node_mut(&mut self, node: NodeId) -> Option<&mut Node> {
        self.slots
            .get_mut(node.index())
            .and_then(|slot| slot.node.as_mut())
    }

    /// Nests the nodes into the tree a layout pass walks.
    ///
    /// The nodes leave the slots: while the tree is out, the host owns no node
    /// and the engine must not mutate it. [`disassemble`](Self::disassemble)
    /// puts the tree back. Returns `None` when the tree is already out.
    pub fn assemble(&mut self) -> Option<Node> {
        if self.assembled {
            return None;
        }
        let tree = self.take_tree(self.container)?;
        self.assembled = true;
        Some(tree)
    }

    /// Puts an assembled tree back into the slots.
    ///
    /// A call while the nodes are already in the host does nothing.
    pub fn disassemble(&mut self, tree: Node) {
        if !self.assembled {
            return;
        }
        self.put_tree(self.container, tree);
        self.assembled = false;
    }

    /// Whether the node of `id` owns the subtree under it.
    fn is_opaque(&self, id: NodeId) -> bool {
        self.slots
            .get(id.index())
            .is_some_and(|slot| slot.opaque)
    }

    /// Takes the subtree of `id` out of the slots, childless nodes nested.
    fn take_tree(&mut self, id: NodeId) -> Option<Node> {
        if self.is_opaque(id) {
            let slot = self.slots.get_mut(id.index())?;
            debug_assert!(
                slot.children.is_empty(),
                "an opaque node owns its subtree and takes no children"
            );
            return slot.node.take();
        }
        let (node, ids) = {
            let slot = self.slots.get_mut(id.index())?;
            (slot.node.take()?, mem::take(&mut slot.children))
        };
        let mut children = Vec::with_capacity(ids.len());
        for child in &ids {
            if let Some(child) = self.take_tree(*child) {
                children.push(child);
            }
        }
        if let Some(slot) = self.slots.get_mut(id.index()) {
            slot.children = ids;
        }
        Some(node.with_children(children))
    }

    /// Returns a node and its subtree to the slots.
    fn put_tree(&mut self, id: NodeId, mut node: Node) {
        if self.is_opaque(id) {
            if let Some(slot) = self.slots.get_mut(id.index()) {
                slot.node = Some(node);
            }
            return;
        }
        let children = node.take_children();
        let ids = match self.slots.get_mut(id.index()) {
            Some(slot) => {
                slot.node = Some(node);
                mem::take(&mut slot.children)
            }
            None => return,
        };
        debug_assert_eq!(children.len(), ids.len());
        let mut children = children.into_iter();
        for child in &ids {
            if let Some(node) = children.next() {
                self.put_tree(*child, node);
            }
        }
        if let Some(slot) = self.slots.get_mut(id.index()) {
            slot.children = ids;
        }
    }

    /// Allocates a slot for `node`.
    fn alloc(&mut self, node: Node, opaque: bool) -> NodeId {
        if let Some(id) = self.free.pop() {
            if let Some(slot) = self.slots.get_mut(id.index()) {
                slot.node = Some(node);
                slot.parent = None;
                slot.children.clear();
                slot.opaque = opaque;
            }
            return id;
        }
        let id = NodeId(self.slots.len() as u32);
        self.slots.push(Slot {
            node: Some(node),
            parent: None,
            children: Vec::new(),
            opaque,
        });
        id
    }
}

impl Default for WidgetHost {
    fn default() -> Self {
        Self::new()
    }
}

impl Host for WidgetHost {
    type Element = NodeElement;
    type Node = NodeId;

    fn element_matches(&self, old: &NodeElement, new: &NodeElement) -> bool {
        old.kind == new.kind
    }

    fn create(&mut self, element: &NodeElement) -> NodeId {
        debug_assert!(!self.assembled, "the host must be disassembled to mutate it");
        self.alloc(element.build(), element.opaque)
    }

    fn update(&mut self, node: NodeId, element: &NodeElement) {
        debug_assert!(!self.assembled, "the host must be disassembled to mutate it");
        // The children belong to the fiber tree, not to the description: the
        // slot keeps them and only the widget and its chain are replaced. An
        // opaque node replaces its own subtree, which is why the element
        // keeps the flag on every update.
        if let Some(slot) = self.slots.get_mut(node.index()) {
            slot.node = Some(element.build());
            slot.opaque = element.opaque;
        }
    }

    fn insert_before(&mut self, parent: NodeId, child: NodeId, anchor: Option<NodeId>) {
        debug_assert!(!self.assembled, "the host must be disassembled to mutate it");
        debug_assert!(
            !self.is_opaque(parent),
            "an opaque node owns its subtree and takes no children"
        );
        if self.is_opaque(parent) {
            return;
        }
        if let Some(current) = self
            .slots
            .get(child.index())
            .and_then(|slot| slot.parent)
            && let Some(slot) = self.slots.get_mut(current.index())
        {
            slot.children.retain(|id| *id != child);
        }
        let Some(slot) = self.slots.get_mut(parent.index()) else {
            return;
        };
        let position = anchor
            .and_then(|anchor| slot.children.iter().position(|id| *id == anchor))
            .unwrap_or(slot.children.len());
        slot.children.insert(position, child);
        if let Some(slot) = self.slots.get_mut(child.index()) {
            slot.parent = Some(parent);
        }
    }

    fn remove(&mut self, parent: NodeId, child: NodeId) {
        debug_assert!(!self.assembled, "the host must be disassembled to mutate it");
        let Some(slot) = self.slots.get_mut(parent.index()) else {
            return;
        };
        slot.children.retain(|id| *id != child);
        let Some(slot) = self.slots.get_mut(child.index()) else {
            return;
        };
        slot.node = None;
        slot.parent = None;
        slot.children.clear();
        self.free.push(child);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::geometry::{Constraints, Rect, Size};
    use crate::theme::Theme;
    use crate::widget::{DrawScope, MeasureScope};

    /// A widget with a fixed size, so a layout can be asserted.
    struct Block {
        side: f32,
    }

    impl Widget for Block {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::new(self.side, self.side))
        }

        fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
            scope.fill_rect(bounds, Color::RED);
        }
    }

    /// A widget of a different kind, for the mismatch tests.
    struct Other;

    impl Widget for Other {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::new(2.0, 2.0))
        }
    }

    fn block(side: f32) -> NodeElement {
        NodeElement::new(move || Block { side })
    }

    /// The marker of the opaque elements of the tests.
    struct Spliced;

    /// An opaque element whose node holds one `Block` child.
    fn spliced(side: f32) -> NodeElement {
        NodeElement::opaque::<Spliced>(move || {
            Node::new(Block { side }).with_children(vec![Node::new(Block { side: side * 0.5 })])
        })
    }

    /// A host with `children` inserted under the container.
    fn host_with(children: &[NodeElement]) -> (WidgetHost, Vec<NodeId>) {
        let mut host = WidgetHost::new();
        let mut ids = Vec::new();
        for element in children {
            let id = host.create(element);
            host.insert_before(host.container(), id, None);
            ids.push(id);
        }
        (host, ids)
    }

    #[test]
    fn an_element_reuses_a_node_of_the_same_widget_kind() {
        let host = WidgetHost::new();
        assert!(host.element_matches(&block(1.0), &block(2.0)));
        assert!(!host.element_matches(&block(1.0), &NodeElement::new(|| Other)));
    }

    #[test]
    fn an_inserted_node_keeps_the_order_the_engine_asks_for() {
        let (mut host, ids) = host_with(&[block(1.0), block(2.0)]);
        assert_eq!(host.children(host.container()), [ids[0], ids[1]]);

        host.insert_before(host.container(), ids[1], Some(ids[0]));
        assert_eq!(host.children(host.container()), [ids[1], ids[0]]);
    }

    #[test]
    fn a_node_moves_out_of_its_old_parent() {
        let (mut host, ids) = host_with(&[block(1.0), block(2.0)]);
        host.insert_before(ids[0], ids[1], None);
        assert_eq!(host.children(host.container()), [ids[0]]);
        assert_eq!(host.children(ids[0]), [ids[1]]);
    }

    #[test]
    fn a_removed_node_leaves_the_tree_and_frees_its_slot() {
        let (mut host, ids) = host_with(&[block(1.0), block(2.0)]);
        host.remove(host.container(), ids[1]);
        assert_eq!(host.children(host.container()), [ids[0]]);
        assert!(host.node(ids[1]).is_none());

        let reused = host.create(&block(3.0));
        assert_eq!(reused, ids[1]);
    }

    #[test]
    fn an_update_replaces_the_widget_and_keeps_the_children() {
        let (mut host, ids) = host_with(&[block(1.0), block(2.0)]);
        host.insert_before(ids[0], ids[1], None);
        host.update(ids[0], &block(5.0));

        let theme = Theme::light();
        let side = host
            .node_mut(ids[0])
            .map(|node| node.measure(&theme, Constraints::unbounded()));
        assert_eq!(side, Some(Size::new(5.0, 5.0)));
        assert_eq!(host.children(ids[0]), [ids[1]]);
    }

    #[test]
    fn assembly_nests_the_nodes_and_disassembly_restores_the_host() {
        let (mut host, ids) = host_with(&[block(1.0), block(2.0)]);
        host.insert_before(ids[0], ids[1], None);

        let tree = host.assemble();
        let Some(tree) = tree else {
            return;
        };
        assert!(host.node(host.container()).is_none());
        assert_eq!(tree.children().len(), 1);
        assert_eq!(tree.children()[0].children().len(), 1);

        host.disassemble(tree);
        assert_eq!(host.children(host.container()), [ids[0]]);
        assert_eq!(host.children(ids[0]), [ids[1]]);
        assert!(host.node(ids[1]).is_some());
    }

    #[test]
    fn an_opaque_node_keeps_the_subtree_its_factory_built() {
        let (mut host, ids) = host_with(&[spliced(4.0)]);
        assert_eq!(host.children(host.container()), [ids[0]]);
        assert!(host.children(ids[0]).is_empty());

        let Some(mut tree) = host.assemble() else {
            return;
        };
        assert_eq!(tree.children()[0].children().len(), 1);

        let theme = Theme::light();
        let size = tree.measure(&theme, Constraints::unbounded());
        tree.place(Rect::new(crate::geometry::Offset::ZERO, size));
        assert_eq!(tree.children()[0].bounds().size, Size::new(4.0, 4.0));
        host.disassemble(tree);
        assert_eq!(
            host.node(ids[0])
                .map(|node| node.children().len()),
            Some(1)
        );
    }

    #[test]
    fn an_opaque_element_matches_only_its_own_marker() {
        struct OtherSubtree;
        let host = WidgetHost::new();
        let first = spliced(1.0);
        let second = spliced(2.0);
        let other = NodeElement::opaque::<OtherSubtree>(|| Node::new(Block { side: 3.0 }));
        assert!(host.element_matches(&first, &second));
        assert!(!host.element_matches(&first, &other));
        assert!(!host.element_matches(&first, &block(1.0)));
    }

    #[test]
    fn an_update_rebuilds_an_opaque_node_and_its_subtree() {
        let (mut host, ids) = host_with(&[spliced(2.0)]);
        host.update(ids[0], &spliced(6.0));

        let Some(mut tree) = host.assemble() else {
            return;
        };
        let theme = Theme::light();
        let size = tree.measure(&theme, Constraints::unbounded());
        assert_eq!(size, Size::new(6.0, 6.0));
        assert_eq!(tree.children()[0].children().len(), 1);
    }

    #[test]
    fn the_container_fills_the_constraints_with_the_root_element() {
        let (mut host, _) = host_with(&[block(7.0)]);
        let Some(mut tree) = host.assemble() else {
            return;
        };
        let theme = Theme::light();
        let size = tree.measure(&theme, Constraints::tight(Size::new(20.0, 15.0)));
        assert_eq!(size, Size::new(20.0, 15.0));
        tree.place(Rect::new(crate::geometry::Offset::ZERO, size));
        assert_eq!(tree.children()[0].bounds().size, Size::new(20.0, 15.0));
    }
}
