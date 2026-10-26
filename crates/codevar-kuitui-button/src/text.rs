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

//! The text buttons: [`TextButton`] and [`ColorableTextButton`].
//!
//! Compose spells these as functions that take a `content` lambda whose
//! only job is to draw a label with the content color of the button.
//! Kuitui has no implicit local to hand that color down, so the label is
//! a widget of its own — [`Label`] — which resolves the content color
//! from the palette and the flags of the button and paints the text with
//! it.
//!
//! The label wraps either of the two text widgets of the framework:
//! [`codevar_kuitui::Text`] for the ordinary run, and
//! [`codevar_kuitui::ColorableText`] when the glyphs must keep the colors
//! of a color font. Both take a [`codevar_kuitui::TextContent`], so a
//! label is multi-span and multi-line, and every character the theme's
//! font covers is drawable.

use alloc::rc::Rc;
use alloc::vec::Vec;

use codevar_kuitui::text::TextContent;
use codevar_kuitui::{
    Color, ColorScheme, ColorableText, Constraints, CornerRadii, DrawScope, EdgeInsets, InteractionSource,
    MeasureScope, Modifier, Node, Rect, Size, Text, Theme, Widget,
};

use crate::button::{Button, Click};
use crate::defaults;
use crate::state::{BorderStroke, ButtonColors, ButtonElevation, ButtonFlags, ButtonPalette};

/// The text widget a [`Label`] can paint.
///
/// The trait exists so that the label does not have to know which of the
/// two text widgets it holds: both are built the same way, from the
/// content of the button and the color the palette resolved.
pub trait LabelText: Widget + 'static {
    /// The widget that draws `content` with `color`.
    fn composed(content: &TextContent, color: Color) -> Self;
}

impl LabelText for Text {
    fn composed(content: &TextContent, color: Color) -> Self {
        Self::with_content(content.clone()).color(color)
    }
}

impl LabelText for ColorableText {
    fn composed(content: &TextContent, color: Color) -> Self {
        Self::with_content(content.clone()).color(color)
    }
}

/// The label of a button: a text widget painted with the content color
/// of that button.
///
/// The color comes from the palette rather than from the theme's
/// `on_surface` because a button tints its own content — the primary on a
/// filled button, the disabled content on one that answers nothing.
/// Material keeps the content color constant under the transient states,
/// so only the flags and the theme can change it; the label rebuilds the
/// widget it wraps when either of them does, and never on a hover.
#[derive(Debug, Clone)]
pub struct Label<W: LabelText> {
    /// The text of the label.
    content: TextContent,
    /// The palette the content color is resolved from.
    palette: ButtonPalette,
    /// What the button does.
    flags: ButtonFlags,
    /// The widget the label paints.
    widget: Option<W>,
    /// The color `widget` was built for.
    painted: Option<Color>,
}

impl<W: LabelText> Label<W> {
    /// The label that draws `content` with the colors of `palette`.
    #[must_use]
    pub fn new(content: TextContent, palette: ButtonPalette, flags: ButtonFlags) -> Self {
        Self {
            content,
            palette,
            flags,
            widget: None,
            painted: None,
        }
    }

    /// The text of the label.
    #[must_use]
    pub const fn content(&self) -> &TextContent {
        &self.content
    }

    /// The content color the label draws with in `scheme`.
    #[must_use]
    pub fn color(&self, scheme: &ColorScheme) -> Color {
        self.palette
            .resolve(scheme)
            .content_for(self.flags)
    }

    /// The wrapped widget, rebuilt when the theme changed the color.
    fn sync(&mut self, theme: &Theme) {
        let color = self.color(&theme.colors);
        if self.painted == Some(color) && self.widget.is_some() {
            return;
        }
        self.widget = Some(W::composed(&self.content, color));
        self.painted = Some(color);
    }
}

impl<W: LabelText> Widget for Label<W> {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        self.sync(scope.theme());
        match &mut self.widget {
            Some(widget) => widget.measure(scope, constraints),
            None => constraints.enforce(Size::ZERO),
        }
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        self.sync(scope.theme());
        if let Some(widget) = &mut self.widget {
            widget.draw(scope, bounds);
        }
    }
}

/// A button whose content is a line of text, the counterpart of Compose's
/// `TextButton`.
///
/// The container of a text button is transparent: what the palette does
/// paint is the content and the state layer a hover leaves behind, which
/// is why [`Button::colors`] defaults to [`ButtonColors::text`] here
/// rather than to the filled role.
///
/// [`ButtonColors::text`]: crate::state::ButtonColors::text
#[derive(Clone)]
pub struct TextButton {
    /// The handler the release calls.
    handler: Click,
    /// The decoration outside the container.
    decoration: Modifier,
    /// The interaction states of the button.
    source: InteractionSource,
    /// What the button does.
    flags: ButtonFlags,
    /// The corners of the container.
    shape: CornerRadii,
    /// The colors of the container, the state layer and the content.
    palette: ButtonPalette,
    /// The ambient shadow, or `None`.
    elevation: Option<ButtonElevation>,
    /// The outline, or `None`.
    border: Option<BorderStroke>,
    /// The space between the container and the label.
    padding: EdgeInsets,
    /// The text of the label.
    label: TextContent,
}

impl TextButton {
    /// The button that calls `on_click` and draws `label`.
    #[must_use]
    pub fn new(on_click: impl Fn() + 'static, label: impl AsRef<str>) -> Self {
        Self::with_content(on_click, TextContent::new(label))
    }

    /// The button that calls `on_click` and draws the multi-span `label`.
    #[must_use]
    pub fn with_content(on_click: impl Fn() + 'static, label: TextContent) -> Self {
        Self {
            handler: Rc::new(on_click),
            decoration: Modifier::none(),
            source: InteractionSource::new(),
            flags: ButtonFlags::STANDARD,
            shape: defaults::SHAPE,
            palette: ButtonPalette::role_of(ButtonColors::text),
            elevation: Some(ButtonElevation::flat()),
            border: None,
            padding: defaults::CONTENT_PADDING,
            label,
        }
    }

    /// The button with `modifier` outside its container.
    #[must_use]
    pub fn modifier(mut self, modifier: Modifier) -> Self {
        self.decoration = modifier;
        self
    }

    /// The button that does what `flags` says.
    #[must_use]
    pub fn flags(mut self, flags: ButtonFlags) -> Self {
        self.flags = flags;
        self
    }

    /// The button with the corners of `shape`.
    #[must_use]
    pub fn shape(mut self, shape: CornerRadii) -> Self {
        self.shape = shape;
        self
    }

    /// The button drawn with `palette`.
    #[must_use]
    pub fn colors(mut self, palette: ButtonPalette) -> Self {
        self.palette = palette;
        self
    }

    /// The button with `elevation` of ambient shadow.
    #[must_use]
    pub fn elevation(mut self, elevation: Option<ButtonElevation>) -> Self {
        self.elevation = elevation;
        self
    }

    /// The button with `border` around its container.
    #[must_use]
    pub fn border(mut self, border: Option<BorderStroke>) -> Self {
        self.border = border;
        self
    }

    /// The button with `padding` between the container and the label.
    #[must_use]
    pub fn content_padding(mut self, padding: EdgeInsets) -> Self {
        self.padding = padding;
        self
    }

    /// The button that reports its states to `source`.
    #[must_use]
    pub fn interaction_source(mut self, source: InteractionSource) -> Self {
        self.source = source;
        self
    }

    /// The label of this button, drawn with its own colors.
    fn label_node<W: LabelText>(&self) -> Node {
        Node::new(Label::<W>::new(self.label.clone(), self.palette, self.flags))
    }
}

impl Button for TextButton {
    fn on_click(&self) -> &Click {
        &self.handler
    }

    fn modifier(&self) -> Modifier {
        self.decoration.clone()
    }

    fn flags(&self) -> ButtonFlags {
        self.flags
    }

    fn shape(&self) -> CornerRadii {
        self.shape
    }

    fn colors(&self) -> ButtonPalette {
        self.palette
    }

    fn elevation(&self) -> Option<ButtonElevation> {
        self.elevation
    }

    fn border(&self) -> Option<BorderStroke> {
        self.border
    }

    fn content_padding(&self) -> EdgeInsets {
        self.padding
    }

    fn interaction_source(&self) -> InteractionSource {
        self.source.clone()
    }

    fn content(&self) -> Vec<Node> {
        alloc::vec![self.label_node::<Text>()]
    }
}

/// A text button whose label keeps the colors of a color font, the
/// counterpart of [`TextButton`] for emoji and other color glyphs.
///
/// The only thing it changes is which of the two text widgets paints the
/// content, so the record the backend replays is a color run instead of
/// an outline run. The content color of the palette is the fallback a
/// backend without a color font draws with.
#[derive(Clone)]
pub struct ColorableTextButton {
    /// The parameters the button answers with.
    button: TextButton,
}

impl ColorableTextButton {
    /// The button that calls `on_click` and draws `label`.
    #[must_use]
    pub fn new(on_click: impl Fn() + 'static, label: impl AsRef<str>) -> Self {
        Self {
            button: TextButton::new(on_click, label),
        }
    }

    /// The button that calls `on_click` and draws the multi-span `label`.
    #[must_use]
    pub fn with_content(on_click: impl Fn() + 'static, label: TextContent) -> Self {
        Self {
            button: TextButton::with_content(on_click, label),
        }
    }

    /// The button with `modifier` outside its container.
    #[must_use]
    pub fn modifier(self, modifier: Modifier) -> Self {
        Self {
            button: self.button.modifier(modifier),
        }
    }

    /// The button that does what `flags` says.
    #[must_use]
    pub fn flags(self, flags: ButtonFlags) -> Self {
        Self {
            button: self.button.flags(flags),
        }
    }

    /// The button with the corners of `shape`.
    #[must_use]
    pub fn shape(self, shape: CornerRadii) -> Self {
        Self {
            button: self.button.shape(shape),
        }
    }

    /// The button drawn with `palette`.
    #[must_use]
    pub fn colors(self, palette: ButtonPalette) -> Self {
        Self {
            button: self.button.colors(palette),
        }
    }

    /// The button with `elevation` of ambient shadow.
    #[must_use]
    pub fn elevation(self, elevation: Option<ButtonElevation>) -> Self {
        Self {
            button: self.button.elevation(elevation),
        }
    }

    /// The button with `border` around its container.
    #[must_use]
    pub fn border(self, border: Option<BorderStroke>) -> Self {
        Self {
            button: self.button.border(border),
        }
    }

    /// The button with `padding` between the container and the label.
    #[must_use]
    pub fn content_padding(self, padding: EdgeInsets) -> Self {
        Self {
            button: self.button.content_padding(padding),
        }
    }

    /// The button that reports its states to `source`.
    #[must_use]
    pub fn interaction_source(self, source: InteractionSource) -> Self {
        Self {
            button: self.button.interaction_source(source),
        }
    }
}

impl Button for ColorableTextButton {
    // The parameters are read from the fields of the wrapped button: the
    // builder methods of `TextButton` have the same names Compose gives
    // its parameters, and a builder takes the argument the getter does
    // not.
    fn on_click(&self) -> &Click {
        &self.button.handler
    }

    fn modifier(&self) -> Modifier {
        self.button.decoration.clone()
    }

    fn flags(&self) -> ButtonFlags {
        self.button.flags
    }

    fn shape(&self) -> CornerRadii {
        self.button.shape
    }

    fn colors(&self) -> ButtonPalette {
        self.button.palette
    }

    fn elevation(&self) -> Option<ButtonElevation> {
        self.button.elevation
    }

    fn border(&self) -> Option<BorderStroke> {
        self.button.border
    }

    fn content_padding(&self) -> EdgeInsets {
        self.button.padding
    }

    fn interaction_source(&self) -> InteractionSource {
        self.button.source.clone()
    }

    fn content(&self) -> Vec<Node> {
        alloc::vec![self.button.label_node::<ColorableText>()]
    }
}
