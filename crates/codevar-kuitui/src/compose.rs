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

//! The composition: the tree, the values it remembers and the
//! recomposition.
//!
//! A composition owns a tree of [`Node`]s and the *content* closure
//! that builds it. Re-running the closure *recomposes* the tree.
//!
//! # Remembering
//!
//! A value that must outlive a recomposition is created with
//! [`Composer::remember`]. The slots are matched by position, exactly
//! as Compose matches them, so a `remember` call must stay at the same
//! place in the content closure from one recomposition to the next.
//!
//! ```
//! use codevar_kuitui::prelude::*;
//!
//! // The counter survives a recomposition even though the closure
//! // that reads it is run again.
//! let mut composition = Composition::new(Theme::light());
//! composition.set_content(|composer| {
//!     let _count = composer.state(0_u32);
//!     kuitui! { Text("hello") }
//! });
//! ```
//!
//! # Dirty tracking
//!
//! A state created with [`Composer::state`] is observed by the
//! composition. Writing to it marks the composition dirty, and
//! [`Composition::recompose`] rebuilds the tree.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::any::Any;

use crate::event::{Event, EventResult};
use crate::geometry::{Constraints, Offset, Rect, Size};
use crate::node::{Node, dispatch_event};
use crate::paint::DrawList;
use crate::state::{DirtyState, MutableState};
use crate::theme::Theme;

/// A remembered value.
enum Slot {
    /// The slot holds a value.
    Value(Box<dyn Any>),
}

/// The values a composition remembers between two recompositions.
struct CompositionState {
    /// The remembered values, matched by position.
    slots: Vec<Slot>,
    /// The states the composition observes.
    observed: Vec<Rc<dyn DirtyState>>,
}

impl CompositionState {
    /// Creates an empty state.
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            observed: Vec::new(),
        }
    }

    /// Whether one of the observed states was written.
    fn is_dirty(&self) -> bool {
        self.observed.iter().any(|state| state.is_dirty())
    }

    /// Marks every observed state as clean.
    fn clear_dirty(&mut self) {
        for state in &self.observed {
            state.clear_dirty();
        }
    }
}

/// Builds the tree of a composition and remembers the values that
/// must outlive a recomposition.
///
/// The composer is handed to the content closure of a composition. It
/// hands out the remembered values and keeps track of the states the
/// tree reads.
pub struct Composer<'a> {
    /// The values remembered so far.
    state: &'a mut CompositionState,
    /// The position of the next slot.
    cursor: usize,
}

impl Composer<'_> {
    /// Returns the value remembered at the current position.
    ///
    /// The value is created by `init` the first time the slot is
    /// reached and returned unchanged afterwards. Because the slots
    /// are matched by position, a `remember` call must stay at the
    /// same place in the content closure from one recomposition to
    /// the next.
    ///
    /// The value must be cloneable: the composer hands a copy of the
    /// remembered value to the caller, so that two widgets may hold
    /// the same state without sharing a mutable borrow of it.
    pub fn remember<T: Clone + 'static>(&mut self, init: impl FnOnce() -> T) -> T {
        let index = self.cursor;
        self.cursor += 1;
        if let Some(Slot::Value(value)) = self.state.slots.get(index)
            && let Some(typed) = value.downcast_ref::<T>()
        {
            return typed.clone();
        }
        let value = init();
        let stored = Box::new(value.clone());
        if index < self.state.slots.len() {
            self.state.slots[index] = Slot::Value(stored);
        } else {
            self.state.slots.push(Slot::Value(stored));
        }
        value
    }

    /// Returns a mutable state that survives a recomposition.
    ///
    /// The state is created with `initial` the first time the slot is
    /// reached. Writing to it marks the composition dirty, so the next
    /// frame recomposes the tree and the widget that reads the state
    /// sees the new value.
    pub fn state<T: 'static>(&mut self, initial: T) -> MutableState<T> {
        let state = self.remember(|| MutableState::new(initial));
        self.observe(&state);
        state
    }

    /// Adds a state to the states the composition watches.
    ///
    /// A state that is not observed does not make the composition
    /// dirty, which is what an application wants for a value it only
    /// reads.
    pub fn observe<T: 'static>(&mut self, state: &MutableState<T>) {
        self.state
            .observed
            .push(Rc::new(MutableState::clone(state)) as Rc<dyn DirtyState>);
    }
}

/// What a recomposition did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Recomposition {
    /// The tree was rebuilt.
    Recomposed,
    /// Nothing was dirty, so the tree was left alone.
    Skipped,
}

/// The closure that builds the tree of a composition.
pub type Content<'a> = Box<dyn FnMut(&mut Composer<'_>) -> Node + 'a>;

/// A composed tree of widgets.
///
/// A composition owns the tree, the values it remembers, the theme it
/// is drawn with and the focus. The application drives it: it
/// recomposes when a state changed, lays the tree out when the window
/// changed size and renders the tree into a display list.
pub struct Composition<'a> {
    /// The content closure that builds the tree.
    content: Content<'a>,
    /// The tree.
    root: Node,
    /// The values remembered between two recompositions.
    state: CompositionState,
    /// The theme the tree is drawn with.
    theme: Theme,
    /// The path of the focused widget, if any.
    focus: Option<Vec<usize>>,
    /// The size the tree was laid out at.
    size: Size,
    /// Whether the tree must be laid out again.
    layout_dirty: bool,
}

impl<'a> Composition<'a> {
    /// Creates an empty composition with the given theme.
    ///
    /// The composition has no content until [`Composition::set_content`]
    /// is called.
    #[must_use]
    pub fn new(theme: Theme) -> Self {
        Self {
            content: Box::new(|_| Node::new(EmptyWidget)),
            root: Node::new(EmptyWidget),
            state: CompositionState::new(),
            theme,
            focus: None,
            size: Size::ZERO,
            layout_dirty: true,
        }
    }

    /// Sets the content of the composition.
    ///
    /// The closure is run by [`Composition::recompose`]. It receives a
    /// [`Composer`] and returns the root of the tree.
    pub fn set_content(&mut self, content: impl FnMut(&mut Composer<'_>) -> Node + 'a) {
        self.content = Box::new(content);
        self.layout_dirty = true;
    }

    /// The theme the tree is drawn with.
    #[inline]
    #[must_use]
    pub const fn theme(&self) -> &Theme {
        &self.theme
    }

    /// Replaces the theme the tree is drawn with.
    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
        self.layout_dirty = true;
    }

    /// The root of the tree.
    #[inline]
    #[must_use]
    pub const fn root(&self) -> &Node {
        &self.root
    }

    /// The root of the tree, for mutation.
    #[inline]
    pub fn root_mut(&mut self) -> &mut Node {
        &mut self.root
    }

    /// The size the tree was laid out at.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// The path of the focused widget, if any.
    #[inline]
    #[must_use]
    pub fn focus(&self) -> Option<&[usize]> {
        self.focus.as_deref()
    }

    /// Gives the focus to the widget at `path`.
    pub fn set_focus(&mut self, path: Option<Vec<usize>>) {
        self.focus = path;
    }

    /// Whether one of the observed states was written since the last
    /// recomposition.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.state.is_dirty()
    }

    /// Rebuilds the tree.
    ///
    /// The content closure is run again and the tree it returns
    /// replaces the old one. The values the composer remembered are
    /// kept, so a state read by the content survives the
    /// recomposition.
    ///
    /// The composition is left alone when no observed state was
    /// written, unless `force` asks for a rebuild anyway.
    pub fn recompose(&mut self, force: bool) -> Recomposition {
        if !force && !self.state.is_dirty() {
            return Recomposition::Skipped;
        }
        let Composition {
            content, root, state, ..
        } = self;
        let mut composer = Composer { state, cursor: 0 };
        *root = content(&mut composer);
        self.state.clear_dirty();
        self.layout_dirty = true;
        Recomposition::Recomposed
    }

    /// Measures and places the tree inside `size`.
    ///
    /// The layout is skipped when nothing changed since the last call
    /// and `force` is `false`.
    pub fn layout(&mut self, size: Size, force: bool) {
        if !force && !self.layout_dirty && self.size == size {
            return;
        }
        self.size = size;
        self.layout_dirty = false;
        let theme = &self.theme;
        let root = &mut self.root;
        root.measure(theme, Constraints::tight(size));
        root.place(Rect::new(Offset::ZERO, size));
    }

    /// Draws the tree into `list`.
    ///
    /// The list is emptied first, so it holds exactly the operations of
    /// this frame.
    pub fn render(&mut self, list: &mut DrawList) {
        list.clear();
        let theme = &self.theme;
        let root = &mut self.root;
        root.draw(theme, list);
    }

    /// Offers an event to the tree.
    ///
    /// The event is offered to the deepest child that contains it; the
    /// result says whether it was consumed.
    pub fn handle_event(&mut self, event: &Event) -> EventResult {
        let theme = &self.theme;
        let root = &mut self.root;
        let mut path = Vec::new();
        dispatch_event(root, theme, &mut path, &mut self.focus, event)
    }
}

/// The widget of an empty composition.
struct EmptyWidget;

impl crate::widget::Widget for EmptyWidget {
    fn measure(&mut self, _scope: &mut crate::widget::MeasureScope<'_>, constraints: Constraints) -> Size {
        constraints.enforce(Size::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::widget::{DrawScope, MeasureScope, Widget};

    /// A widget that paints a rectangle of a given size.
    struct Block {
        width: f32,
        height: f32,
        color: Color,
    }

    impl Widget for Block {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            constraints.enforce(Size::new(self.width, self.height))
        }

        fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
            scope.fill_rect(bounds, self.color);
        }
    }

    #[test]
    fn a_remembered_value_survives_a_recomposition() {
        let mut composition = Composition::new(Theme::light());
        let initializations = Rc::new(core::cell::Cell::new(0_u32));
        let observed = Rc::clone(&initializations);
        composition.set_content(move |composer| {
            let count = composer.remember(|| {
                observed.set(observed.get() + 1);
                7_u32
            });
            Node::new(Block {
                width: count as f32,
                height: 1.0,
                color: Color::RED,
            })
        });
        composition.recompose(true);
        composition.recompose(true);
        assert_eq!(initializations.get(), 1);
    }

    #[test]
    fn a_tight_layout_fills_the_space_it_is_given() {
        let mut composition = Composition::new(Theme::light());
        composition.set_content(|_| {
            Node::new(Block {
                width: 4.0,
                height: 4.0,
                color: Color::RED,
            })
        });
        composition.recompose(true);
        composition.layout(Size::new(10.0, 10.0), true);
        assert_eq!(composition.root().size(), Size::new(10.0, 10.0));
    }

    #[test]
    fn a_state_makes_the_composition_dirty() {
        let mut composition = Composition::new(Theme::light());
        let state: Rc<MutableState<u32>> = Rc::new(MutableState::new(0));
        let observed = Rc::clone(&state);
        composition.set_content(move |composer| {
            composer.observe(&observed);
            Node::new(Block {
                width: 1.0,
                height: 1.0,
                color: Color::BLACK,
            })
        });
        composition.recompose(true);
        assert!(!composition.is_dirty());
        state.set(1);
        assert!(composition.is_dirty());
    }

    #[test]
    fn a_composition_that_is_not_dirty_is_not_recomposed() {
        let mut composition = Composition::new(Theme::light());
        composition.set_content(|_| {
            Node::new(Block {
                width: 1.0,
                height: 1.0,
                color: Color::BLACK,
            })
        });
        composition.recompose(true);
        assert_eq!(composition.recompose(false), Recomposition::Skipped);
        assert_eq!(composition.recompose(true), Recomposition::Recomposed);
    }

    #[test]
    fn rendering_fills_the_display_list() {
        let mut composition = Composition::new(Theme::light());
        composition.set_content(|_| {
            Node::new(Block {
                width: 4.0,
                height: 4.0,
                color: Color::BLUE,
            })
        });
        composition.recompose(true);
        composition.layout(Size::new(20.0, 20.0), true);
        let mut list = DrawList::new();
        composition.render(&mut list);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn rendering_twice_does_not_double_the_operations() {
        let mut composition = Composition::new(Theme::light());
        composition.set_content(|_| {
            Node::new(Block {
                width: 4.0,
                height: 4.0,
                color: Color::BLUE,
            })
        });
        composition.recompose(true);
        composition.layout(Size::new(20.0, 20.0), true);
        let mut list = DrawList::new();
        composition.render(&mut list);
        composition.render(&mut list);
        assert_eq!(list.len(), 1);
    }
}
