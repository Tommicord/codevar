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

//! The reactive composition: the engine driving the widget tree.
//!
//! [`ReactiveComposition`] is the composition that runs on
//! `codevar-kuitui-rea`. The application hands it content — a function from the
//! render scope to an element tree — and the engine turns that tree into
//! fibers, keeps the hooks alive across renders and commits the host mutations
//! a re-render needs. What the application keeps from
//! [`Composition`](crate::compose::Composition) is the frame: the same layout,
//! paint, event and focus calls, in the same order.
//!
//! # The content is a component
//!
//! The content closure is rendered as the root component. It receives a
//! [`RenderCtx`], so it reads state with
//! [`use_state`](RenderCtx::use_state) and describes the tree with
//! [`Element::host`] leaves; a nested component is
//! [`Element::component`], and both keep their hooks as long as the parent
//! renders the same source expression. The composition wraps the content
//! itself, so swapping it with
//! [`set_content`](ReactiveComposition::set_content) keeps the root component's
//! hooks instead of remounting them.
//!
//! # The tree comes and goes
//!
//! The host stores the widget nodes childless and nests them into the tree a
//! layout pass walks only while the application uses it. A recomposition takes
//! the tree apart, runs the engine — which creates, updates, moves and removes
//! nodes in the host — and nests the result again. The swap is invisible from
//! outside: every method but
//! [`recompose`](ReactiveComposition::recompose) and
//! [`flush_passive`](ReactiveComposition::flush_passive) sees the assembled
//! tree.
//!
//! ```
//! use codevar_kuitui::prelude::*;
//! use codevar_kuitui::rea::Element;
//!
//! let mut composition = ReactiveComposition::new(Theme::light());
//! composition.set_content(|ctx| {
//!     let count = ctx.use_state(|| 0_u32);
//!     Element::host(NodeElement::new(move || Text::new(count.get().to_string())))
//! });
//! composition.recompose(true);
//! composition.layout(Size::new(200.0, 40.0), true);
//! ```
//!
//! # Effects
//!
//! A commit whose lanes are discrete runs its passive effects before it
//! returns, so a state written by an event is observable in the same frame. A
//! transition or idle commit leaves them pending, and the application runs
//! them with [`flush_passive`](ReactiveComposition::flush_passive) once the
//! frame that produced them is presented — the equivalent of React's post-paint
//! effect flush.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::mem;

use codevar_kuitui_rea::{Element, RenderCtx, Root};

use crate::compose::Recomposition;
use crate::event::{Event, EventResult};
use crate::geometry::{Constraints, Offset, Rect, Size};
use crate::host::WidgetHost;
use crate::node::{Node, dispatch_event};
use crate::paint::DrawList;
use crate::theme::Theme;
use crate::widget::Widget;

/// The function that builds the element tree of a reactive composition.
pub type Content = Rc<dyn Fn(&mut RenderCtx<'_>) -> Element<WidgetHost>>;

/// A composition driven by the reactivity engine.
///
/// It owns the root of the fiber tree, the theme the tree is drawn with, the
/// focus and the assembled widget tree. An application drives it the same way
/// it drives [`Composition`](crate::compose::Composition): recompose when
/// something changed, lay the tree out when the window changed size, render it
/// into a display list and offer it the events.
pub struct ReactiveComposition {
    /// The function that builds the element tree.
    content: Content,
    /// The mounted tree, with its host, fibers and lanes.
    root: Root<WidgetHost>,
    /// The assembled widget tree.
    tree: Node,
    /// The theme the tree is drawn with.
    theme: Theme,
    /// The path of the focused widget, if any.
    focus: Option<Vec<usize>>,
    /// The size the tree was laid out at.
    size: Size,
    /// Whether the tree must be laid out again.
    layout_dirty: bool,
    /// Whether the content must be rendered as the root element again.
    content_dirty: bool,
}

impl ReactiveComposition {
    /// Creates an empty composition with the given theme.
    ///
    /// The composition has no content until
    /// [`set_content`](Self::set_content) is called.
    #[must_use]
    pub fn new(theme: Theme) -> Self {
        let host = WidgetHost::new();
        let container = host.container();
        Self {
            content: Rc::new(|_| Element::empty()),
            root: Root::new(host, container),
            tree: Node::new(Placeholder),
            theme,
            focus: None,
            size: Size::ZERO,
            layout_dirty: true,
            content_dirty: false,
        }
    }

    /// Sets the content of the composition.
    ///
    /// The closure is run as the root component by
    /// [`recompose`](Self::recompose). It receives the render scope of that
    /// component, so the hooks it calls are the root's and survive a later
    /// replacement of the content.
    pub fn set_content(&mut self, content: impl Fn(&mut RenderCtx<'_>) -> Element<WidgetHost> + 'static) {
        self.content = Rc::new(content);
        self.content_dirty = true;
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

    /// The root of the assembled tree.
    #[inline]
    #[must_use]
    pub const fn root(&self) -> &Node {
        &self.tree
    }

    /// The host the nodes live in.
    ///
    /// The host holds no node while the tree is assembled, but its structure —
    /// the parent and the child order of every node — is always readable.
    #[inline]
    #[must_use]
    pub fn host(&self) -> &WidgetHost {
        self.root.host()
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

    /// Whether a render is waiting to be applied.
    ///
    /// The composition is dirty when the content was replaced or when a state
    /// a component reads was written since the last recomposition.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.content_dirty || self.root.has_pending_work()
    }

    /// Renders every pending update and commits it to the widget tree.
    ///
    /// With `force`, the root element is rendered even when nothing is pending,
    /// which is what an application calls after it changed a value the engine
    /// cannot observe. Otherwise a composition with no pending work is left
    /// alone.
    pub fn recompose(&mut self, force: bool) -> Recomposition {
        if !force && !self.is_dirty() {
            return Recomposition::Skipped;
        }
        let rebuild = force || self.content_dirty;
        let content = Rc::clone(&self.content);
        self.on_host(|root| {
            if rebuild {
                // The wrapper is one source expression, so the root component
                // keeps its identity — and its hooks — across renders.
                root.render(Element::component(move |ctx| content(ctx)));
            }
            root.flush();
        });
        self.content_dirty = false;
        self.layout_dirty = true;
        Recomposition::Recomposed
    }

    /// Runs the passive effects of the last commit.
    ///
    /// A discrete commit has already run its effects by the time
    /// [`recompose`](Self::recompose) returns. This runs the ones a transition
    /// or an idle render left pending, together with the work they schedule.
    /// Returns whether there were effects to run.
    pub fn flush_passive(&mut self) -> bool {
        let flushed = self.on_host(|root| root.flush_passive());
        if flushed {
            self.layout_dirty = true;
        }
        flushed
    }

    /// Measures and places the tree inside `size`.
    ///
    /// The layout is skipped when nothing changed since the last call and
    /// `force` is `false`.
    pub fn layout(&mut self, size: Size, force: bool) {
        if !force && !self.layout_dirty && self.size == size {
            return;
        }
        self.size = size;
        self.layout_dirty = false;
        let theme = &self.theme;
        self.tree.measure(theme, Constraints::tight(size));
        self.tree.place(Rect::new(Offset::ZERO, size));
    }

    /// Draws the tree into `list`.
    ///
    /// The list is emptied first, so it holds exactly the operations of this
    /// frame.
    pub fn render(&mut self, list: &mut DrawList) {
        list.clear();
        self.tree.draw(&self.theme, list);
    }

    /// Offers an event to the tree.
    ///
    /// The event is offered to the deepest child that contains it; the result
    /// says whether it was consumed.
    pub fn handle_event(&mut self, event: &Event) -> EventResult {
        let mut path = Vec::new();
        dispatch_event(&mut self.tree, &self.theme, &mut path, &mut self.focus, event)
    }

    /// Runs `work` with the nodes back in the host.
    ///
    /// The engine mutates the host, so the tree has to leave the composition
    /// while it runs; no public method runs in that window, so the placeholder
    /// the swap installs is never used. The host always has the tree to give
    /// back, and a miss would leave the nodes in the host for the next
    /// assembly rather than lose them.
    fn on_host<R>(&mut self, work: impl FnOnce(&mut Root<WidgetHost>) -> R) -> R {
        let tree = mem::replace(&mut self.tree, Node::new(Placeholder));
        self.root.host_mut().disassemble(tree);
        let result = work(&mut self.root);
        if let Some(tree) = self.root.host_mut().assemble() {
            self.tree = tree;
        }
        result
    }
}

/// The node the composition holds while the host owns the tree.
struct Placeholder;

impl Widget for Placeholder {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::cell::{Cell, RefCell};

    use codevar_kuitui_rea::{Lanes, State};

    use crate::color::Color;
    use crate::event::{PointerEvent, PointerEventKind};
    use crate::host::NodeElement;
    use crate::node::Modifier;
    use crate::widget::{DrawScope, MeasureScope};

    /// A widget that records the side it was built with when it is measured,
    /// so a test can see which description reached the tree.
    struct Probe {
        /// The side the widget reports.
        side: f32,
        /// The side the last measurement saw.
        seen: Rc<Cell<f32>>,
    }

    impl Widget for Probe {
        fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
            self.seen.set(self.side);
            constraints.enforce(Size::new(self.side, self.side))
        }
    }

    /// A widget of a fixed size that paints its whole box.
    struct Block {
        /// The side the widget reports.
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

    /// An element holding a [`Probe`] of `side`.
    fn probe(side: f32, seen: &Rc<Cell<f32>>) -> Element<WidgetHost> {
        let seen = Rc::clone(seen);
        Element::host(NodeElement::new(move || Probe {
            side,
            seen: Rc::clone(&seen),
        }))
    }

    #[test]
    fn the_content_is_mounted_into_the_tree() {
        let seen = Rc::new(Cell::new(0.0_f32));
        let observed = Rc::clone(&seen);
        let mut composition = ReactiveComposition::new(Theme::light());
        composition.set_content(move |_| probe(3.0, &observed));

        assert_eq!(composition.recompose(false), Recomposition::Recomposed);
        composition.layout(Size::new(20.0, 20.0), false);

        assert_eq!(seen.get(), 3.0);
        assert_eq!(composition.root().children().len(), 1);
        assert_eq!(composition.size(), Size::new(20.0, 20.0));
    }

    #[test]
    fn an_unchanged_composition_is_not_recomposed() {
        let mut composition = ReactiveComposition::new(Theme::light());
        composition.set_content(|_| Element::empty());

        assert_eq!(composition.recompose(false), Recomposition::Recomposed);
        assert!(!composition.is_dirty());
        assert_eq!(composition.recompose(false), Recomposition::Skipped);
        assert_eq!(composition.recompose(true), Recomposition::Recomposed);
    }

    #[test]
    fn a_state_update_reaches_the_node_that_reads_it() {
        let seen = Rc::new(Cell::new(0.0_f32));
        let observed = Rc::clone(&seen);
        let handle: Rc<RefCell<Option<State<u32>>>> = Rc::new(RefCell::new(None));
        let slot = Rc::clone(&handle);
        let mut composition = ReactiveComposition::new(Theme::light());
        composition.set_content(move |ctx| {
            let side = ctx.use_state(|| 1_u32);
            *slot.borrow_mut() = Some(side.clone());
            probe(*side.get() as f32, &observed)
        });
        composition.recompose(false);
        composition.layout(Size::new(10.0, 10.0), false);
        assert_eq!(seen.get(), 1.0);

        let Some(side) = handle.borrow().clone() else {
            return;
        };
        side.set(2);
        assert!(composition.is_dirty());
        assert_eq!(composition.recompose(false), Recomposition::Recomposed);
        composition.layout(Size::new(10.0, 10.0), false);
        assert_eq!(seen.get(), 2.0);
    }

    #[test]
    fn a_keyed_reorder_moves_the_same_nodes() {
        let flipped = Rc::new(Cell::new(false));
        let flag = Rc::clone(&flipped);
        let seen = Rc::new(Cell::new(0.0_f32));
        let observed = Rc::clone(&seen);
        let mut composition = ReactiveComposition::new(Theme::light());
        composition.set_content(move |_| {
            let first = probe(1.0, &observed).keyed("a");
            let second = probe(1.0, &observed).keyed("b");
            if flag.get() {
                Element::fragment([second, first])
            } else {
                Element::fragment([first, second])
            }
        });
        composition.recompose(false);
        let before: Vec<_> = composition
            .host()
            .children(composition.host().container())
            .to_vec();
        assert_eq!(before.len(), 2);

        flipped.set(true);
        composition.recompose(true);
        let after = composition
            .host()
            .children(composition.host().container())
            .to_vec();
        assert_eq!(after, [before[1], before[0]]);
    }

    #[test]
    fn a_passive_effect_waits_for_the_presentation() {
        let runs = Rc::new(Cell::new(0_u32));
        let counter = Rc::clone(&runs);
        let handle: Rc<RefCell<Option<State<u32>>>> = Rc::new(RefCell::new(None));
        let slot = Rc::clone(&handle);
        let seen = Rc::new(Cell::new(0.0_f32));
        let observed = Rc::clone(&seen);
        let mut composition = ReactiveComposition::new(Theme::light());
        composition.set_content(move |ctx| {
            let value = ctx.use_state(|| 0_u32);
            *slot.borrow_mut() = Some(value.clone());
            let counter = Rc::clone(&counter);
            let rendered = *value.get();
            ctx.use_effect(rendered, move || {
                counter.set(counter.get() + 1);
                None
            });
            probe(rendered as f32, &observed)
        });
        composition.recompose(false);
        assert_eq!(runs.get(), 1);

        let Some(value) = handle.borrow().clone() else {
            return;
        };
        let Some(lane) = Lanes::from_index(4) else {
            return;
        };
        value.set_with_lane(lane, 1);
        assert_eq!(composition.recompose(false), Recomposition::Recomposed);
        assert_eq!(runs.get(), 1);
        assert!(composition.flush_passive());
        assert_eq!(runs.get(), 2);
    }

    #[test]
    fn an_event_reaches_the_widget_tree() {
        let clicks = Rc::new(Cell::new(0_u32));
        let counter = Rc::clone(&clicks);
        let mut composition = ReactiveComposition::new(Theme::light());
        composition.set_content(move |_| {
            let counter = Rc::clone(&counter);
            Element::host(
                NodeElement::new(|| Block { side: 40.0 }).modifier(Modifier::none().clickable(
                    move |_event| {
                        counter.set(counter.get() + 1);
                        EventResult::Consumed
                    },
                )),
            )
        });
        composition.recompose(false);
        composition.layout(Size::new(100.0, 100.0), false);

        let down = Event::Pointer(PointerEvent::new(PointerEventKind::Down, Offset::new(5.0, 5.0)));
        assert!(composition.handle_event(&down).is_consumed());
        assert_eq!(clicks.get(), 1);
    }

    #[test]
    fn rendering_fills_the_display_list() {
        let mut composition = ReactiveComposition::new(Theme::light());
        composition.set_content(|_| Element::host(NodeElement::new(|| Block { side: 4.0 })));
        composition.recompose(false);
        composition.layout(Size::new(20.0, 20.0), false);

        let mut list = DrawList::new();
        composition.render(&mut list);
        assert_eq!(list.len(), 1);
    }
}
