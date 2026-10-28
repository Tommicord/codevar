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

//! The element syntax of the reactive composition.
//!
//! [`kuitui!`](crate::kuitui) builds a [`Node`](crate::node::Node) tree for
//! [`Composition`](crate::compose::Composition), which owns its widget tree and
//! rebuilds it on every recomposition. [`element!`] builds the same tree as
//! elements for
//! [`ReactiveComposition`](crate::reactive::ReactiveComposition): every widget
//! becomes a host element, a container carries its children as elements, and
//! the engine keeps the nodes and fibers across a render.
//!
//! ```
//! use codevar_kuitui::prelude::*;
//!
//! let mut composition = ReactiveComposition::new(Theme::light());
//! composition.set_content(|ctx| {
//!     let count = ctx.use_state(|| 0_u32);
//!     element! {
//!         Column(spacing: 4.0) {
//!             Text(count.get().to_string())
//!         }
//!     }
//! });
//! composition.recompose(true);
//! ```
//!
//! # The shape of the syntax
//!
//! An item is a widget name, an argument list and an optional block of
//! children, exactly as in [`kuitui!`](crate::kuitui): the named arguments are
//! the builder methods, and `modifier:` decorates the node. A subtree built
//! outside the macro is spliced in with `Element(expr)`, which takes any
//! expression of type
//! [`Element<WidgetHost>`](codevar_kuitui_rea::Element).
//!
//! # Identity
//!
//! A host element is identified by its widget type, so a `Text` keeps its node
//! when the text it draws changes, and a `Column` that becomes a `Row`
//! replaces it. The macro clones the widget it built into the factory the
//! engine calls, which is why the widget types it accepts are `Clone`; a
//! widget that is not is described with
//! [`NodeElement::new`](crate::host::NodeElement::new) directly.

/// Builds a tree of elements for the reactive composition.
///
/// The macro accepts a widget name, an argument list and an optional block of
/// children, in the same syntax as [`kuitui!`](crate::kuitui):
///
/// ```
/// use codevar_kuitui::prelude::*;
///
/// let tree = element! {
///     Column(spacing: 8.0) {
///         Text("Codevar")
///         Spacer(height: 12.0)
///     }
/// };
/// ```
///
/// A subtree built by another library is spliced in with `Element(expr)`:
///
/// ```
/// use codevar_kuitui::prelude::*;
///
/// let label = element! { Text("Save") };
/// let tree = element! {
///     Row {
///         Element(label)
///         Spacer(width: 4.0)
///     }
/// };
/// ```
#[macro_export]
macro_rules! element {
    (Element ( $($element:tt)* )) => {{
        $($element)*
    }};
    ($name:ident (modifier : $modifier:expr $(, $arg:ident : $value:expr)* $(,)?)
     $( { $($child:tt)* } )?) => {{
        $crate::elements::private::host_with_modifier(
            $crate::widgets::$name::new()$(.$arg($value))*,
            $crate::__kuitui_element_children!($($($child)*)?),
            $modifier,
        )
    }};
    ($name:ident ($value:expr) $( { $($child:tt)* } )?) => {{
        $crate::elements::private::host(
            $crate::widgets::$name::new($value),
            $crate::__kuitui_element_children!($($($child)*)?),
        )
    }};
    ($name:ident ($value:expr, $($arg:ident : $named:expr),+ $(,)?) $( { $($child:tt)* } )?) => {{
        $crate::elements::private::host(
            $crate::widgets::$name::new($value)$(.$arg($named))*,
            $crate::__kuitui_element_children!($($($child)*)?),
        )
    }};
    ($name:ident { $($child:tt)* }) => {{
        $crate::elements::private::host(
            $crate::widgets::$name::new(),
            $crate::__kuitui_element_children!($($child)*),
        )
    }};
    ($name:ident ($($arg:ident : $value:expr),* $(,)?) $( { $($child:tt)* } )?) => {{
        $crate::elements::private::host(
            $crate::widgets::$name::new()$(.$arg($value))*,
            $crate::__kuitui_element_children!($($($child)*)?),
        )
    }};
}

/// Builds the list of the children of an item.
///
/// The macro is an implementation detail of [`element!`] and is not meant to
/// be called directly.
#[doc(hidden)]
#[macro_export]
macro_rules! __kuitui_element_children {
    () => {
        $crate::elements::private::no_children()
    };
    (Element ( $($element:tt)* ) $($rest:tt)*) => {{
        let mut __children = $crate::elements::private::no_children();
        __children.push($($element)*);
        $crate::elements::private::append(&mut __children, $crate::__kuitui_element_children!($($rest)*));
        __children
    }};
    ($name:ident { $($inner:tt)* } $($rest:tt)*) => {{
        let mut __children = $crate::elements::private::no_children();
        __children.push($crate::element!($name { $($inner)* }));
        $crate::elements::private::append(&mut __children, $crate::__kuitui_element_children!($($rest)*));
        __children
    }};
    ($name:ident ( $($args:tt)* ) { $($inner:tt)* } $($rest:tt)*) => {{
        let mut __children = $crate::elements::private::no_children();
        __children.push($crate::element!($name ( $($args)* ) { $($inner)* }));
        $crate::elements::private::append(&mut __children, $crate::__kuitui_element_children!($($rest)*));
        __children
    }};
    ($name:ident ( $($args:tt)* ) $($rest:tt)*) => {{
        let mut __children = $crate::elements::private::no_children();
        __children.push($crate::element!($name ( $($args)* )));
        $crate::elements::private::append(&mut __children, $crate::__kuitui_element_children!($($rest)*));
        __children
    }};
}

/// The helpers [`element!`] expands to.
///
/// The module is an implementation detail and may change at any time.
#[doc(hidden)]
pub mod private {
    use alloc::vec::Vec;

    use codevar_kuitui_rea::Element;

    use crate::host::{NodeElement, WidgetHost};
    use crate::node::Modifier;
    use crate::widget::Widget;

    /// Creates the element of `widget` with `children`.
    ///
    /// The factory clones the widget, so the host rebuilds the node from the
    /// description the render produced instead of the one it was created with.
    pub fn host<W: Widget + Clone + 'static>(
        widget: W,
        children: Vec<Element<WidgetHost>>,
    ) -> Element<WidgetHost> {
        Element::host(NodeElement::new(move || widget.clone())).children(children)
    }

    /// Creates the element of `widget` with `children`, decorated with
    /// `modifier`.
    pub fn host_with_modifier<W: Widget + Clone + 'static>(
        widget: W,
        children: Vec<Element<WidgetHost>>,
        modifier: Modifier,
    ) -> Element<WidgetHost> {
        Element::host(NodeElement::new(move || widget.clone()).modifier(modifier)).children(children)
    }

    /// Creates an empty list of children.
    pub fn no_children() -> Vec<Element<WidgetHost>> {
        Vec::new()
    }

    /// Appends the children of a nested list to the list of their parent.
    pub fn append(target: &mut Vec<Element<WidgetHost>>, source: Vec<Element<WidgetHost>>) {
        target.extend(source);
    }
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;
    use core::cell::Cell;

    use codevar_kuitui_rea::Element;

    use crate::event::{Event, EventResult, PointerEvent, PointerEventKind};
    use crate::geometry::{Offset, Size};
    use crate::host::WidgetHost;
    use crate::node::Modifier;
    use crate::reactive::ReactiveComposition;
    use crate::theme::Theme;

    /// Mounts the element the content builds and returns the composition.
    fn composed(content: impl Fn() -> Element<WidgetHost> + 'static) -> ReactiveComposition {
        let mut composition = ReactiveComposition::new(Theme::light());
        composition.set_content(move |_| content());
        composition.recompose(true);
        composition
    }

    #[test]
    fn the_macro_builds_a_container_with_its_children() {
        let composition = composed(|| {
            element! {
                Column(spacing: 8.0) {
                    Text("one")
                    Text("two")
                }
            }
        });
        let host = composition.host();
        let roots = host.children(host.container());
        assert_eq!(roots.len(), 1);
        assert_eq!(host.children(roots[0]).len(), 2);
    }

    #[test]
    fn the_macro_accepts_a_positional_argument_and_splices_an_element() {
        let label = element! { Text("one") };
        let composition = composed(move || {
            element! {
                Row {
                    Element(label.clone())
                    Text("two")
                }
            }
        });
        let host = composition.host();
        let roots = host.children(host.container());
        assert_eq!(host.children(roots[0]).len(), 2);
    }

    #[test]
    fn a_widget_keeps_its_node_when_the_content_changes() {
        let side = Rc::new(Cell::new(4.0_f32));
        let read = Rc::clone(&side);
        let mut composition = ReactiveComposition::new(Theme::light());
        composition.set_content(move |_| {
            element! {
                Column {
                    Text(if read.get() > 0.0 { "a" } else { "b" })
                }
            }
        });
        composition.recompose(true);
        let host = composition.host();
        let roots = host.children(host.container());
        let label = host.children(roots[0])[0];

        side.set(0.0);
        composition.recompose(true);
        let host = composition.host();
        let roots = host.children(host.container());
        assert_eq!(host.children(roots[0])[0], label);
    }

    #[test]
    fn the_modifier_decorates_the_node_the_macro_builds() {
        let clicks = Rc::new(Cell::new(0_u32));
        let holder = Rc::clone(&clicks);
        let mut composition = ReactiveComposition::new(Theme::light());
        composition.set_content(move |_| {
            let counted = Rc::clone(&holder);
            element! {
                Column(modifier: Modifier::none().clickable(move |event| {
                    if matches!(event, Event::Pointer(pointer) if pointer.kind == PointerEventKind::Down) {
                        counted.set(counted.get() + 1);
                        return EventResult::Consumed;
                    }
                    EventResult::Ignored
                })) {
                    Spacer(height: 8.0)
                }
            }
        });
        composition.recompose(true);
        composition.layout(Size::new(20.0, 20.0), true);

        let down = Event::Pointer(PointerEvent::new(PointerEventKind::Down, Offset::new(5.0, 5.0)));
        assert!(composition.handle_event(&down).is_consumed());
        assert_eq!(clicks.get(), 1);
    }
}
