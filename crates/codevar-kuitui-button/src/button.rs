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

//! The button trait and the container widget that answers to it.
//!
//! [`Button`] carries the parameters of Compose's `Button`, and
//! [`Button::node`] turns them into the node of the tree: a
//! [`ButtonChrome`] that paints the container and lays the content out
//! in a centered row, decorated with the modifier the caller asked for
//! plus the tracking the flags ask for.
//!
//! The container is drawn by the widget, not by the modifier chain,
//! because the colors of a button depend on the interaction states and
//! on the theme, and neither is known when the modifier is built. The
//! chrome reads them at draw time, so a press changes one rectangle and
//! nothing else.

use alloc::rc::Rc;
use alloc::vec::Vec;

use codevar_kuitui::{
    Color, ColorScheme, Constraints, CornerRadii, DrawScope, EdgeInsets, Effect, Event, EventResult,
    EventScope, InteractionSource, InteractionState, MeasureScope, Modifier, Node, Offset, PlaceScope,
    PointerEventKind, Rect, Size, Widget,
};

use crate::defaults;
use crate::state::{BorderStroke, ButtonColors, ButtonElevation, ButtonFlags, ButtonPalette};

/// The closure a button calls when a pointer is released inside it.
///
/// Compose's `onClick: () -> Unit` is a lambda the composition rebuilds
/// every frame; Kuitui keeps it behind an `Rc` because the handler
/// travels into the modifier chain, where the framework calls it for as
/// long as the node lives.
pub type Click = Rc<dyn Fn()>;

/// A container the user clicks.
///
/// The methods are the parameters of Jetpack Compose's `Button`, in the
/// order Compose declares them, with the two booleans of the original
/// (`enabled`, and the hover tracking an interactive composable always
/// has) folded into [`Button::flags`]. `contentPadding` is the padding
/// *of the chrome* rather than of the modifier chain: the container of a
/// button wraps its content, so the space belongs to the widget that
/// draws the container.
///
/// # Examples
///
/// ```
/// use codevar_kuitui::{InteractionSource, Node, Text};
/// use codevar_kuitui_button::button::{Button, Click};
/// use std::rc::Rc;
///
/// /// A button that saves: it owns its handler and its states.
/// struct SaveButton {
///     handler: Click,
///     source: InteractionSource,
/// }
///
/// impl SaveButton {
///     fn new(on_save: impl Fn() + 'static) -> Self {
///         Self {
///             handler: Rc::new(on_save),
///             source: InteractionSource::new(),
///         }
///     }
/// }
///
/// impl Button for SaveButton {
///     fn on_click(&self) -> &Click {
///         &self.handler
///     }
///
///     fn interaction_source(&self) -> InteractionSource {
///         self.source.clone()
///     }
///
///     fn content(&self) -> Vec<Node> {
///         vec![Node::new(Text::new("Save"))]
///     }
/// }
///
/// let node = SaveButton::new(|| {}).node();
/// ```
pub trait Button {
    /// The closure called when a pointer button is released inside the
    /// button.
    fn on_click(&self) -> &Click;

    /// The decoration the application asked for.
    ///
    /// The button adds its own operations after these, so a `size` or a
    /// `background` written here decorates the outer box of the button.
    fn modifier(&self) -> Modifier {
        Modifier::none()
    }

    /// What the button does.
    fn flags(&self) -> ButtonFlags {
        ButtonFlags::STANDARD
    }

    /// The corners of the container.
    fn shape(&self) -> CornerRadii {
        defaults::SHAPE
    }

    /// The colors of the container, the state layer and the content.
    ///
    /// The default is the filled role of whichever theme the tree is
    /// drawn with, which is what Compose's `Button` draws.
    fn colors(&self) -> ButtonPalette {
        ButtonPalette::role_of(ButtonColors::filled)
    }

    /// The ambient shadow of the button, or `None` for a button that
    /// sits flat on the surface.
    fn elevation(&self) -> Option<ButtonElevation> {
        Some(ButtonElevation::flat())
    }

    /// The outline of the button, or `None`.
    fn border(&self) -> Option<BorderStroke> {
        None
    }

    /// The space between the container and the content.
    fn content_padding(&self) -> EdgeInsets {
        defaults::CONTENT_PADDING
    }

    /// The interaction states of the button.
    ///
    /// This is Compose's `interactionSource`, and it is the one
    /// parameter an application has to `remember`: Kuitui keeps the tree
    /// between two frames, so a source built per frame forgets where the
    /// pointer is the moment it moves. An implementation returns a clone
    /// of the handle it owns.
    fn interaction_source(&self) -> InteractionSource;

    /// The children of the content, in the order they are laid out.
    fn content(&self) -> Vec<Node>;

    /// The node that draws the button.
    fn node(&self) -> Node {
        button_node(self)
    }
}

/// Builds the node of `button`: the chrome, its content and the modifier
/// that tracks the pointer and, for a button with corners, cuts the painting
/// of the node to the shape of its container.
///
/// A free function so that it answers to a `&dyn Button` too, whose own
/// [`Button::node`] cannot be reached without a sized `Self`.
#[must_use]
pub fn button_node<B: Button + ?Sized>(button: &B) -> Node {
    let flags = button.flags();
    let source = button.interaction_source();
    let chrome = ButtonChrome::new(
        button.colors(),
        flags,
        button.shape(),
        button.elevation(),
        button.border(),
        button.content_padding(),
        source.clone(),
    );
    let mut modifier = button.modifier();
    if flags.contains(ButtonFlags::HOVERABLE) {
        modifier = modifier.hoverable(source.clone());
    }
    if flags.contains(ButtonFlags::ENABLED) {
        let click = Rc::clone(button.on_click());
        if flags.contains(ButtonFlags::SELECTABLE) {
            // The framework flips the selection before it calls this
            // handler, so the click is reported with the new state
            // already written to the source.
            modifier = modifier.selectable(source, move |_event, _selected| {
                (click)();
                EventResult::Consumed
            });
        } else {
            modifier = modifier.clickable_with(source, move |event| {
                if matches!(event, Event::Pointer(pointer) if pointer.kind == PointerEventKind::Up) {
                    (click)();
                    return EventResult::Consumed;
                }
                EventResult::Ignored
            });
        }
    }
    // A canvas that paints the whole node box cannot know the shape of the
    // button it decorates, so its square corners would cover the rounding
    // of the container. The node clips its own painting to that shape; with
    // square corners there is nothing to cut, and the content keeps the free
    // overflow a rectangular button has always had.
    let shape = button.shape();
    let modifier = if shape.is_zero() {
        modifier
    } else {
        modifier.clip_rounded(shape)
    };
    Node::new(chrome)
        .with_children(button.content())
        .modifier(modifier)
}

/// The container of a button: the shadow, the rounded box, the state
/// layer, the outline and the row of content inside them.
///
/// The widget owns the content padding because Compose measures a button
/// around its content: the padding is inside the container, so the
/// widget that draws the container is the one that has to know it.
///
/// Everything the chrome paints belongs to the background pass of the
/// node, because the container is opaque: painted in `draw` it would
/// cover the decoration a [`Modifier::concrete`] canvas draws over the
/// node and the ramp a [`Modifier::gradient`] names for it, which is
/// what made a decorated button look undecorated.
#[derive(Debug, Clone)]
pub struct ButtonChrome {
    /// The colors of the container and the content.
    palette: ButtonPalette,
    /// What the button does.
    flags: ButtonFlags,
    /// The corners of the container.
    shape: CornerRadii,
    /// The ambient shadow, if the button has one.
    elevation: Option<ButtonElevation>,
    /// The outline, if the button has one.
    border: Option<BorderStroke>,
    /// The space between the container and the content.
    padding: EdgeInsets,
    /// The space between two children of the content.
    spacing: f32,
    /// The states the colors are resolved from.
    source: InteractionSource,
}

impl ButtonChrome {
    /// The chrome of a button with the given parameters.
    #[must_use]
    pub fn new(
        palette: ButtonPalette,
        flags: ButtonFlags,
        shape: CornerRadii,
        elevation: Option<ButtonElevation>,
        border: Option<BorderStroke>,
        padding: EdgeInsets,
        source: InteractionSource,
    ) -> Self {
        Self {
            palette,
            flags,
            shape,
            elevation,
            border,
            padding,
            spacing: defaults::CONTENT_SPACING,
            source,
        }
    }

    /// The chrome with `spacing` between two children of the content.
    #[must_use]
    pub fn spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    /// The states the pointer left on the button.
    #[must_use]
    pub fn state(&self) -> InteractionState {
        self.source.state()
    }

    /// The colors the chrome draws with in `scheme`.
    #[must_use]
    pub fn colors(&self, scheme: &ColorScheme) -> ButtonColors {
        self.palette.resolve(scheme)
    }

    /// The content area `bounds` leaves once the padding is taken, in
    /// the coordinate space of the widget.
    fn content_area(&self, bounds: Rect) -> Rect {
        Rect::new(
            Offset::new(self.padding.left, self.padding.top),
            Size::new(
                (bounds.size.width - self.padding.horizontal()).max(0.0),
                (bounds.size.height - self.padding.vertical()).max(0.0),
            ),
        )
    }
}

impl Widget for ButtonChrome {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let padding = self.padding;
        let max_width = (constraints.max_width - padding.horizontal()).max(0.0);
        // The content is measured loose, like the Box Compose wraps a
        // button's content in: a wide container stretches the button,
        // not the label inside it. The height stays unbounded so a run
        // taller than the padded area reports its whole line box — a
        // clamped measurement hides the overflow from `place`, which
        // then gives the run the size of the area and top-aligns it.
        let inner = Constraints::new(0.0, max_width, 0.0, f32::INFINITY);
        let mut width = 0.0_f32;
        let mut height = 0.0_f32;
        let count = scope.child_count();
        for index in 0..count {
            let size = scope.measure_child(index, inner);
            if index > 0 {
                width += self.spacing;
            }
            width += size.width;
            height = height.max(size.height);
        }
        constraints.enforce(Size::new(
            width + padding.horizontal(),
            height + padding.vertical(),
        ))
    }

    fn place(&mut self, scope: &mut PlaceScope<'_>, bounds: Rect) {
        let area = self.content_area(bounds);
        let count = scope.child_count();
        let mut row_width = 0.0_f32;
        let mut row_height = 0.0_f32;
        for index in 0..count {
            let size = scope.child_size(index);
            if index > 0 {
                row_width += self.spacing;
            }
            row_width += size.width;
            row_height = row_height.max(size.height);
        }
        // The row of content is centered in the padded area and every
        // child is centered in the row, which is Compose's
        // `spacedBy(8.dp, CenterHorizontally)` inside a box aligned to
        // its center. A row wider or taller than the area overflows it
        // on both sides: clamping the offset to zero would top-align an
        // oversized run against the padding and push its ink below the
        // center of the container.
        let mut x = area.left() + (area.size.width - row_width) * 0.5;
        let top = area.top() + (area.size.height - row_height) * 0.5;
        for index in 0..count {
            let size = scope.child_size(index);
            let y = top + (row_height - size.height) * 0.5;
            // The scope adds the origin of the widget, so the offset is the
            // position of the child in the coordinate space of the chrome.
            scope.place_child(index, Offset::new(x, y));
            x += size.width + self.spacing;
        }
    }

    fn draw_background(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        let colors = self.palette.resolve(&scope.theme().colors);
        let state = self.source.state();
        let container = Rect::new(Offset::ZERO, bounds.size);
        if let Some(elevation) = self.elevation
            && let Some(sigma) = elevation.sigma_for(state, self.flags)
        {
            // The shadow is the blurred silhouette of the container,
            // drawn under it. The layer program evaluates the falloff of
            // the rounded box, so the halo costs one distance per
            // fragment whatever the sigma.
            scope.push_effect(Effect::new(container).blur(sigma, self.shape));
            scope.fill_rounded_rect(container, self.shape, elevation.color);
            scope.pop_effect();
        }
        paint_rounded(
            scope,
            container,
            self.shape,
            colors.container_for(state, self.flags),
        );
        paint_rounded(
            scope,
            container,
            self.shape,
            colors.overlay_for(state, self.flags),
        );
        if let Some(border) = self.border {
            let color = if self.flags.contains(ButtonFlags::ENABLED) {
                border.color
            } else {
                border
                    .color
                    .with_alpha(defaults::DISABLED_OUTLINE_ALPHA)
            };
            scope.stroke_rect(container, border.radii_for(self.shape), color, border.width);
        }
    }

    fn on_event(&mut self, scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        if self.flags.contains(ButtonFlags::FOCUSABLE)
            && matches!(event, Event::Pointer(pointer) if pointer.kind == PointerEventKind::Down)
        {
            scope.request_focus();
            self.source.set(InteractionState::FOCUSED);
        }
        EventResult::Ignored
    }

    fn hit_testable(&self) -> bool {
        self.flags.contains(ButtonFlags::ENABLED)
    }
}

/// Paints `rect` with `color`, asking the layer program for the corners
/// only when there is something to round.
///
/// A fully transparent color is skipped: the outlined and the text role
/// both draw an empty container, and recording a rectangle that cannot
/// be seen would open a shader layer for nothing.
fn paint_rounded(scope: &mut DrawScope<'_>, rect: Rect, radii: CornerRadii, color: Color) {
    if color.alpha <= 0.0 {
        return;
    }
    let effect = Effect::new(rect).shape(radii);
    if effect.is_identity() {
        scope.fill_rect(rect, color);
        return;
    }
    scope.push_effect(effect);
    scope.fill_rounded_rect(rect, radii, color);
    scope.pop_effect();
}

#[cfg(test)]
mod tests {
    use codevar_kuitui::{
        Constraints, EdgeInsets, InteractionSource, Modifier, Node, Offset, Rect, Size, Text, Theme,
    };

    use super::{ButtonChrome, ButtonColors, ButtonFlags, ButtonPalette, CornerRadii};

    /// The chrome of a button with no decoration and `padding`.
    fn chrome(padding: EdgeInsets) -> ButtonChrome {
        ButtonChrome::new(
            ButtonPalette::role_of(ButtonColors::filled),
            ButtonFlags::STANDARD,
            CornerRadii::ZERO,
            None,
            None,
            padding,
            InteractionSource::new(),
        )
    }

    #[test]
    fn a_label_taller_than_the_content_area_is_centered_vertically() {
        let theme = Theme::light();
        let mut node = Node::new(chrome(EdgeInsets::symmetric(0.0, 10.0)))
            .with_children([Node::new(Text::new("x")).modifier(Modifier::none().size(None, Some(60.0)))]);
        let measured = node.measure(&theme, Constraints::new(0.0, 320.0, 0.0, 64.0));
        assert_eq!(measured.height, 64.0);
        assert_eq!(
            node.children()[0].size().height,
            60.0,
            "the content reports its whole box instead of the area it is given"
        );
        node.place(Rect::new(Offset::ZERO, Size::new(320.0, 64.0)));
        let center = node.children()[0].bounds().center();
        assert!(
            (center.y - 32.0).abs() < 1e-4,
            "an overflowing label centers on the container, not on the padding"
        );
    }

    #[test]
    fn a_row_wider_than_the_content_area_is_centered_horizontally() {
        let theme = Theme::light();
        let mut node = Node::new(chrome(EdgeInsets::ZERO))
            .with_children([Node::new(Text::new("x")).modifier(Modifier::none().size(Some(400.0), None))]);
        node.measure(&theme, Constraints::unbounded());
        node.place(Rect::new(Offset::ZERO, Size::new(320.0, 64.0)));
        let center = node.children()[0].bounds().center();
        assert!(
            (center.x - 160.0).abs() < 1e-4,
            "an overflowing row centers on the container, not on its left edge"
        );
    }
}
