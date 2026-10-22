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

//! The Compose-like syntax that builds a tree.
//!
//! [`kuitui!`] builds a tree of widgets with a syntax that reads like
//! the one Compose uses:
//!
//! ```
//! use codevar_kuitui::prelude::*;
//!
//! let tree = kuitui! {
//!     Column(spacing: 8.0) {
//!         Text("Codevar")
//!         Row(spacing: 4.0) {
//!             Spacer(width: 4.0)
//!             Text("Kuitui")
//!         }
//!     }
//! };
//! ```
//!
//! # The shape of the syntax
//!
//! An item is a widget name, an argument list and an optional block
//! of children:
//!
//! ```text
//! Widget
//! Widget(argument)
//! Widget(name: value, other: value)
//! Widget { child, child }
//! ```
//!
//! The named arguments are the builder methods of the widget, so
//! `Column(spacing: 8.0)` calls `Column::new().spacing(8.0)`. The
//! first argument may be `modifier:`, which decorates the node the
//! widget ends up in.

/// Builds a tree of widgets.
///
/// The macro accepts a widget name, an argument list and an optional
/// block of children:
///
/// ```
/// use codevar_kuitui::prelude::*;
///
/// let tree = kuitui! {
///     Column(spacing: 8.0) {
///         Text("Codevar")
///         Spacer(height: 12.0)
///     }
/// };
/// ```
#[macro_export]
macro_rules! kuitui {
    ($name:ident (modifier : $modifier:expr $(, $arg:ident : $value:expr)* $(,)?)
     $( { $($child:tt)* } )?) => {{
        $crate::__private::kuitui_node_with_modifier(
            $crate::widgets::$name::new()$(.$arg($value))*,
            $crate::__kuitui_children!($($($child)*)?),
            $modifier,
        )
    }};
    ($name:ident ($value:expr) $( { $($child:tt)* } )?) => {{
        $crate::__private::kuitui_node(
            $crate::widgets::$name::new($value),
            $crate::__kuitui_children!($($($child)*)?),
        )
    }};
    ($name:ident ($value:expr, $($arg:ident : $named:expr),+ $(,)?) $( { $($child:tt)* } )?) => {{
        $crate::__private::kuitui_node(
            $crate::widgets::$name::new($value)$(.$arg($named))*,
            $crate::__kuitui_children!($($($child)*)?),
        )
    }};
    ($name:ident { $($child:tt)* }) => {{
        $crate::__private::kuitui_node(
            $crate::widgets::$name::new(),
            $crate::__kuitui_children!($($child)*),
        )
    }};
    ($name:ident ($($arg:ident : $value:expr),* $(,)?) $( { $($child:tt)* } )?) => {{
        $crate::__private::kuitui_node(
            $crate::widgets::$name::new()$(.$arg($value))*,
            $crate::__kuitui_children!($($($child)*)?),
        )
    }};
}

/// Builds the list of the children of an item.
///
/// The macro is an implementation detail of [`kuitui!`] and is not
/// meant to be called directly.
#[doc(hidden)]
#[macro_export]
macro_rules! __kuitui_children {
    () => {
        $crate::__private::empty_children()
    };
    ($name:ident { $($inner:tt)* } $($rest:tt)*) => {{
        let mut __children = $crate::__private::empty_children();
        __children.push($crate::kuitui!($name { $($inner)* }));
        $crate::__private::append_children(&mut __children, $crate::__kuitui_children!($($rest)*));
        __children
    }};
    ($name:ident ( $($args:tt)* ) { $($inner:tt)* } $($rest:tt)*) => {{
        let mut __children = $crate::__private::empty_children();
        __children.push($crate::kuitui!($name ( $($args)* ) { $($inner)* }));
        $crate::__private::append_children(&mut __children, $crate::__kuitui_children!($($rest)*));
        __children
    }};
    ($name:ident ( $($args:tt)* ) $($rest:tt)*) => {{
        let mut __children = $crate::__private::empty_children();
        __children.push($crate::kuitui!($name ( $($args)* )));
        $crate::__private::append_children(&mut __children, $crate::__kuitui_children!($($rest)*));
        __children
    }};
}

/// The helpers the macros of the crate expand to.
///
/// The module is an implementation detail and may change at any time.
#[doc(hidden)]
pub mod __private {
    use alloc::vec::Vec;

    use crate::node::{Modifier, Node};
    use crate::widget::Widget;

    /// Creates a node holding `widget` and `children`.
    pub fn kuitui_node(widget: impl Widget + 'static, children: Vec<Node>) -> Node {
        Node::new(widget).with_children(children)
    }

    /// Creates a node holding `widget` and `children`, decorated with
    /// `modifier`.
    pub fn kuitui_node_with_modifier(
        widget: impl Widget + 'static,
        children: Vec<Node>,
        modifier: Modifier,
    ) -> Node {
        Node::new(widget)
            .with_children(children)
            .modifier(modifier)
    }

    /// Creates an empty list of children.
    pub fn empty_children() -> Vec<Node> {
        Vec::new()
    }

    /// Appends the children of a nested list to the list of their
    /// parent.
    pub fn append_children(target: &mut Vec<Node>, source: Vec<Node>) {
        target.extend(source);
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::{Constraints, Offset, Rect, Size};
    use crate::theme::Theme;

    #[test]
    fn the_macro_builds_a_tree_with_the_children_of_the_block() {
        let tree = crate::kuitui! {
            Column(spacing: 8.0) {
                Text("one")
                Text("two")
            }
        };
        assert_eq!(tree.children().len(), 2);
    }

    #[test]
    fn the_macro_accepts_a_positional_argument() {
        let tree = crate::kuitui! {
            Column {
                Text("one")
            }
        };
        assert_eq!(tree.children().len(), 1);
    }

    #[test]
    fn the_macro_builds_a_tree_without_children() {
        let tree = crate::kuitui! {
            Text("alone")
        };
        assert_eq!(tree.children().len(), 0);
    }

    #[test]
    fn the_macro_nests_the_blocks() {
        let tree = crate::kuitui! {
            Column {
                Row {
                    Text("one")
                    Text("two")
                }
                Text("three")
            }
        };
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].children().len(), 2);
    }

    #[test]
    fn the_macro_accepts_a_modifier() {
        let tree = crate::kuitui! {
            Column(modifier: crate::node::Modifier::none().padding(
                crate::geometry::EdgeInsets::all(4.0)
            )) {
                Text("one")
            }
        };
        let theme = Theme::light();
        let mut tree = tree;
        let size = tree.measure(&theme, Constraints::unbounded());
        assert!(size.width > 0.0);
    }

    #[test]
    fn the_macro_places_the_children_in_order() {
        let theme = Theme::light();
        let mut tree = crate::kuitui! {
            Column(spacing: 2.0) {
                Text("one")
                Text("two")
            }
        };
        tree.measure(&theme, Constraints::unbounded());
        tree.place(Rect::new(Offset::ZERO, Size::new(100.0, 100.0)));
        let first = tree.children()[0].bounds();
        let second = tree.children()[1].bounds();
        assert!(second.top() > first.top());
    }
}
