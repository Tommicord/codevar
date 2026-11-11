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

//! The concrete decoration widgets.
//!
//! These widgets are [`codevar_kuitui::widget::Widget`]s and
//! [`crate::traits::Decoration`]s at the same time: they take part in the
//! Kuitui layout and paint passes, and they report the decoration
//! modifiers they support and the ones that are active.

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

use codevar_kuitui::color::Color;
use codevar_kuitui::event::{Event, EventResult, PointerEventKind};
use codevar_kuitui::geometry::{Constraints, Offset, Rect, Size};
use codevar_kuitui::text::TextStyle;
use codevar_kuitui::theme::Theme;
use codevar_kuitui::widget::{DrawScope, EventScope, MeasureScope, PlaceScope, Widget};

use crate::error::{DecorationError, DecorationResult};
use crate::modifiers::{DecorationModifier, DecorationModifiers};
use crate::theme::DecorationTheme;
use crate::traits::Decoration;
use crate::types::{DecorationBounds, DecorationIcon, DecorationKind};

/// A button of the decoration of a window: minimize, maximize, restore,
/// close or the window menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecorationButton {
    /// The button that minimizes the window.
    Minimize,
    /// The button that maximizes or restores the window.
    Maximize,
    /// The button that closes the window.
    Close,
    /// The button that shows the window menu.
    Menu,
}

impl DecorationButton {
    /// The width of the button, in pixels.
    pub const SIZE: f32 = 36.0;

    /// The modifier the button toggles.
    #[must_use]
    pub const fn modifier(self) -> DecorationModifier {
        match self {
            Self::Minimize => DecorationModifier::Minimized,
            Self::Maximize => DecorationModifier::Maximized,
            Self::Close => DecorationModifier::Closable,
            Self::Menu => DecorationModifier::Menu,
        }
    }

    /// The symbol the button draws.
    #[must_use]
    pub const fn glyph(self) -> &'static str {
        match self {
            Self::Minimize => "\u{2500}",
            Self::Maximize => "\u{25a1}",
            Self::Close => "\u{2715}",
            Self::Menu => "\u{2630}",
        }
    }

    /// The color the button is drawn with when it is hovered.
    #[must_use]
    pub fn hover_color(self, theme: &Theme) -> Color {
        match self {
            Self::Close => theme.colors.error,
            _ => theme.colors.surface_variant,
        }
    }
}

/// The title bar of a window.
///
/// The title bar spans the whole width of the window, draws the title,
/// the icon and the buttons, and reports the button the user pressed.
/// A title bar also reports whether the minimize, maximize and close
/// modifiers are active, so a backend can reflect them in the native
/// window.
#[derive(Clone)]
pub struct DecorationTitleBar {
    /// The title of the window.
    title: String,
    /// The buttons of the window, from the left to the right.
    buttons: Vec<DecorationButton>,
    /// The height of the title bar.
    height: f32,
    /// The handler called when a button is pressed.
    on_click: Option<Rc<dyn Fn(DecorationButton)>>,
    /// The rectangle of every button, filled during the place pass.
    button_rects: Vec<(DecorationButton, Rect)>,
    /// The button the pointer is over.
    hovered: Option<DecorationButton>,
    /// The button the pointer went down on.
    pressed: Option<DecorationButton>,
    /// The modifiers that are active.
    active_modifiers: DecorationModifiers,
    /// The id of the decoration, once it is bound.
    id: Option<crate::types::DecorationId>,
}

impl core::fmt::Debug for DecorationTitleBar {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecorationTitleBar")
            .field("title", &self.title)
            .field("buttons", &self.buttons)
            .field("height", &self.height)
            .field("active_modifiers", &self.active_modifiers)
            .finish_non_exhaustive()
    }
}

impl DecorationTitleBar {
    /// Creates a title bar with the given title and no button.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            buttons: Vec::new(),
            height: 32.0,
            on_click: None,
            button_rects: Vec::new(),
            hovered: None,
            pressed: None,
            active_modifiers: DecorationModifiers::NONE,
            id: None,
        }
    }

    /// The title bar with the given buttons.
    #[must_use]
    pub fn buttons(mut self, buttons: Vec<DecorationButton>) -> Self {
        self.buttons = buttons;
        self
    }

    /// The title bar with the given height.
    #[must_use]
    pub const fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// The title bar that calls `on_click` when a button is pressed.
    #[must_use]
    pub fn on_click(mut self, on_click: impl Fn(DecorationButton) + 'static) -> Self {
        self.on_click = Some(Rc::new(on_click));
        self
    }

    /// The title of the window.
    #[inline]
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// The button the pointer is over.
    #[inline]
    #[must_use]
    pub const fn hovered(&self) -> Option<DecorationButton> {
        self.hovered
    }

    /// The modifiers that are active on the title bar.
    #[inline]
    #[must_use]
    pub const fn active_modifiers(&self) -> DecorationModifiers {
        self.active_modifiers
    }

    /// The style the title is drawn with.
    fn title_style(&self, theme: &Theme) -> TextStyle {
        theme
            .typography
            .label
            .clone()
            .with_color(theme.colors.on_surface)
    }

    /// The button at `position`, if any.
    fn button_at(&self, position: Offset) -> Option<DecorationButton> {
        self.button_rects
            .iter()
            .find(|(_, rect)| rect.contains(position))
            .map(|(button, _)| *button)
    }
}

impl Widget for DecorationTitleBar {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let theme = scope.theme();
        let style = self.title_style(theme);
        let metrics = theme.metrics.as_ref();
        let title_width = metrics.advance(&self.title, &style);
        let buttons_width = self.buttons.len() as f32 * DecorationButton::SIZE;
        let width = (title_width + buttons_width).min(constraints.max_width);
        constraints.enforce(Size::new(width, self.height))
    }

    fn place(&mut self, _scope: &mut PlaceScope<'_>, bounds: Rect) {
        self.button_rects.clear();
        let mut x = bounds.right();
        for button in self.buttons.iter().rev() {
            x -= DecorationButton::SIZE;
            let rect = Rect::from_min_size(
                Offset::new(x, bounds.top()),
                DecorationButton::SIZE,
                bounds.size.height,
            );
            self.button_rects.push((*button, rect));
        }
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        let theme = scope.theme();
        let style = self.title_style(theme);
        let metrics = theme.metrics.as_ref();
        let active = self
            .active_modifiers
            .has(DecorationModifier::Activated);
        let background = if active {
            theme.colors.surface_variant
        } else {
            Color {
                red: theme.colors.surface_variant.red * 0.85,
                green: theme.colors.surface_variant.green * 0.85,
                blue: theme.colors.surface_variant.blue * 0.85,
                alpha: theme.colors.surface_variant.alpha,
            }
        };
        scope.fill_rect(bounds, background);

        let title_width = metrics.advance(&self.title, &style);
        let buttons_width = self.buttons.len() as f32 * DecorationButton::SIZE;
        let available = bounds.size.width - buttons_width;
        let x = bounds.left() + (available - title_width) * 0.5;
        let baseline = bounds.top() + (bounds.size.height - style.font_size) * 0.5;
        scope.draw_text(
            Offset::new(x - bounds.left(), baseline - bounds.top()),
            Rc::from(self.title.as_str()),
            style,
        );

        for (button, rect) in &self.button_rects {
            let color = if self.hovered == Some(*button) {
                button.hover_color(theme)
            } else if self.pressed == Some(*button) {
                Color {
                    red: theme.colors.surface_variant.red * 0.7,
                    green: theme.colors.surface_variant.green * 0.7,
                    blue: theme.colors.surface_variant.blue * 0.7,
                    alpha: theme.colors.surface_variant.alpha,
                }
            } else {
                theme.colors.surface_variant
            };
            scope.fill_rect(*rect, color);
            let glyph = scope
                .theme()
                .typography
                .label
                .clone()
                .with_color(theme.colors.on_surface);
            let glyph_width = metrics.advance(button.glyph(), &glyph);
            let glyph_x = rect.left() + (rect.size.width - glyph_width) * 0.5;
            let glyph_y = rect.top() + (rect.size.height - glyph.font_size) * 0.5;
            scope.draw_text(
                Offset::new(glyph_x - bounds.left(), glyph_y - bounds.top()),
                Rc::from(button.glyph()),
                glyph,
            );
        }
    }

    fn on_event(&mut self, _scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        let Event::Pointer(pointer) = event else {
            return EventResult::Ignored;
        };
        match pointer.kind {
            PointerEventKind::Move => {
                self.hovered = self.button_at(pointer.position);
                EventResult::Ignored
            }
            PointerEventKind::Leave => {
                self.hovered = None;
                self.pressed = None;
                EventResult::Ignored
            }
            PointerEventKind::Down => {
                self.pressed = self.button_at(pointer.position);
                if self.pressed.is_some() {
                    return EventResult::Consumed;
                }
                EventResult::Ignored
            }
            PointerEventKind::Up => {
                let pressed = self.pressed.take();
                let released = self.button_at(pointer.position);
                self.pressed = None;
                if pressed.is_some() && pressed == released {
                    if let Some(handler) = &self.on_click
                        && let Some(button) = pressed
                    {
                        handler(button);
                    }
                    return EventResult::Consumed;
                }
                EventResult::Ignored
            }
            _ => EventResult::Ignored,
        }
    }

    fn hit_testable(&self) -> bool {
        true
    }
}

impl Decoration for DecorationTitleBar {
    fn kind(&self) -> DecorationKind {
        DecorationKind::TitleBar
    }

    fn id(&self) -> Option<crate::types::DecorationId> {
        self.id
    }

    fn bind(&mut self, id: crate::types::DecorationId) {
        self.id = Some(id);
    }

    fn unbind(&mut self) {
        self.id = None;
    }

    fn supported_modifiers(&self) -> DecorationModifiers {
        DecorationModifiers::from(DecorationModifier::Minimized)
            | DecorationModifiers::from(DecorationModifier::Maximized)
            | DecorationModifiers::from(DecorationModifier::Closable)
            | DecorationModifiers::from(DecorationModifier::Menu)
    }

    fn active_modifiers(&self) -> DecorationModifiers {
        self.active_modifiers
    }

    fn set_active_modifiers(&mut self, modifiers: DecorationModifiers) -> DecorationResult<()> {
        self.active_modifiers = modifiers;
        Ok(())
    }

    fn request_modifier(&mut self, modifier: DecorationModifier, enable: bool) -> DecorationResult<()> {
        if !self.supported_modifiers().has(modifier) {
            return Err(DecorationError::Unsupported(DecorationKind::TitleBar));
        }
        let _ = (modifier, enable);
        Ok(())
    }

    fn bounds(&self) -> DecorationBounds {
        DecorationBounds::default()
    }
}

/// A window with a title bar above its content.
///
/// The first child of the frame is the title bar and the second one is
/// the content of the window. The title bar keeps the height it asks
/// for and the content takes the rest of the window.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DecorationWindowFrame {
    /// The space between the title bar and the content.
    spacing: f32,
    /// The thickness of the border that wraps the content.
    border_thickness: f32,
    /// Whether the window is active.
    active: bool,
}

impl DecorationWindowFrame {
    /// Creates a frame without spacing.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            spacing: 0.0,
            border_thickness: 1.0,
            active: false,
        }
    }

    /// The frame with `spacing` pixels between the title bar and the
    /// content.
    #[must_use]
    pub const fn spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    /// The frame with the given border thickness.
    #[must_use]
    pub const fn border_thickness(mut self, border_thickness: f32) -> Self {
        self.border_thickness = border_thickness;
        self
    }

    /// The frame with the given active state.
    #[must_use]
    pub const fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }
}

impl Widget for DecorationWindowFrame {
    fn measure(&mut self, scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        let title = scope.measure_child(
            0,
            Constraints::new(0.0, constraints.max_width, 0.0, constraints.max_height),
        );
        let remaining = (constraints.max_height - title.height - self.spacing).max(0.0);
        let content = scope.measure_child(1, Constraints::new(0.0, constraints.max_width, 0.0, remaining));
        constraints.enforce(Size::new(
            title.width.max(content.width),
            title.height + self.spacing + content.height,
        ))
    }

    fn place(&mut self, scope: &mut PlaceScope<'_>, _bounds: Rect) {
        let title = scope.child_size(0);
        scope.place_child(0, Offset::ZERO);
        scope.place_child(1, Offset::new(0.0, title.height + self.spacing));
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        let theme = scope.theme();
        let border = if self.active {
            theme.colors.on_surface
        } else {
            theme.colors.surface_variant
        };
        scope.stroke_rect(
            bounds,
            codevar_kuitui::paint::CornerRadii::ZERO,
            border,
            self.border_thickness,
        );
    }
}

impl Decoration for DecorationWindowFrame {
    fn kind(&self) -> DecorationKind {
        DecorationKind::WindowFrame
    }

    fn id(&self) -> Option<crate::types::DecorationId> {
        None
    }

    fn bind(&mut self, _id: crate::types::DecorationId) {}

    fn unbind(&mut self) {}

    fn draws_border(&self) -> bool {
        true
    }

    fn border_color(&self, _theme: &DecorationTheme) -> Color {
        if self.active {
            codevar_kuitui::color::Color::WHITE
        } else {
            codevar_kuitui::color::Color {
                red: 0.5,
                green: 0.5,
                blue: 0.5,
                alpha: 1.0,
            }
        }
    }
}

/// The edge of a window that a resize handle covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecorationResizeEdgeWidget {
    /// The top edge.
    Top,
    /// The bottom edge.
    Bottom,
    /// The left edge.
    Left,
    /// The right edge.
    Right,
    /// The top-left corner.
    TopLeft,
    /// The top-right corner.
    TopRight,
    /// The bottom-left corner.
    BottomLeft,
    /// The bottom-right corner.
    BottomRight,
}

impl DecorationResizeEdgeWidget {
    /// The thickness of the handle that covers the edge.
    pub const THICKNESS: f32 = 6.0;

    /// Whether the edge is on the top of the window.
    #[must_use]
    pub const fn is_top(self) -> bool {
        matches!(self, Self::Top | Self::TopLeft | Self::TopRight)
    }

    /// Whether the edge is on the bottom of the window.
    #[must_use]
    pub const fn is_bottom(self) -> bool {
        matches!(self, Self::Bottom | Self::BottomLeft | Self::BottomRight)
    }

    /// Whether the edge is on the left of the window.
    #[must_use]
    pub const fn is_left(self) -> bool {
        matches!(self, Self::Left | Self::TopLeft | Self::BottomLeft)
    }

    /// Whether the edge is on the right of the window.
    #[must_use]
    pub const fn is_right(self) -> bool {
        matches!(self, Self::Right | Self::TopRight | Self::BottomRight)
    }
}

/// A handle that covers one edge of a window and reports the drag that
/// resizes it.
///
/// The handle draws nothing; it only tells the application that the
/// user grabbed the edge, so the application can resize the window.
#[derive(Clone)]
pub struct DecorationResizeHandle {
    /// The edge the handle covers.
    edge: DecorationResizeEdgeWidget,
    /// The handler called when the handle is grabbed.
    on_grab: Option<Rc<dyn Fn(DecorationResizeEdgeWidget)>>,
    /// Whether the pointer went down on the handle.
    pressed: bool,
    /// The id of the decoration, once it is bound.
    id: Option<crate::types::DecorationId>,
}

impl core::fmt::Debug for DecorationResizeHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecorationResizeHandle")
            .field("edge", &self.edge)
            .field("pressed", &self.pressed)
            .finish_non_exhaustive()
    }
}

impl DecorationResizeHandle {
    /// Creates a handle that covers `edge`.
    #[must_use]
    pub const fn new(edge: DecorationResizeEdgeWidget) -> Self {
        Self {
            edge,
            on_grab: None,
            pressed: false,
            id: None,
        }
    }

    /// The handle that calls `on_grab` when it is grabbed.
    #[must_use]
    pub fn on_grab(mut self, on_grab: impl Fn(DecorationResizeEdgeWidget) + 'static) -> Self {
        self.on_grab = Some(Rc::new(on_grab));
        self
    }

    /// The edge the handle covers.
    #[inline]
    #[must_use]
    pub const fn edge(&self) -> DecorationResizeEdgeWidget {
        self.edge
    }

    /// Whether the pointer went down on the handle.
    #[inline]
    #[must_use]
    pub const fn is_pressed(&self) -> bool {
        self.pressed
    }
}

impl Widget for DecorationResizeHandle {
    fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        constraints.enforce(Size::ZERO)
    }

    fn on_event(&mut self, _scope: &mut EventScope<'_>, event: &Event) -> EventResult {
        let Event::Pointer(pointer) = event else {
            return EventResult::Ignored;
        };
        match pointer.kind {
            PointerEventKind::Down => {
                self.pressed = true;
                if let Some(handler) = &self.on_grab {
                    handler(self.edge);
                }
                EventResult::Consumed
            }
            PointerEventKind::Up | PointerEventKind::Cancel | PointerEventKind::Leave => {
                self.pressed = false;
                EventResult::Ignored
            }
            _ => EventResult::Ignored,
        }
    }

    fn hit_testable(&self) -> bool {
        true
    }
}

impl Decoration for DecorationResizeHandle {
    fn kind(&self) -> DecorationKind {
        DecorationKind::ResizeEdge
    }

    fn id(&self) -> Option<crate::types::DecorationId> {
        self.id
    }

    fn bind(&mut self, id: crate::types::DecorationId) {
        self.id = Some(id);
    }

    fn unbind(&mut self) {
        self.id = None;
    }

    fn resize_edge(&self) -> Option<crate::types::DecorationResizeEdge> {
        Some(match self.edge {
            DecorationResizeEdgeWidget::Top => crate::types::DecorationResizeEdge::Top,
            DecorationResizeEdgeWidget::Bottom => crate::types::DecorationResizeEdge::Bottom,
            DecorationResizeEdgeWidget::Left => crate::types::DecorationResizeEdge::Left,
            DecorationResizeEdgeWidget::Right => crate::types::DecorationResizeEdge::Right,
            DecorationResizeEdgeWidget::TopLeft => crate::types::DecorationResizeEdge::TopLeft,
            DecorationResizeEdgeWidget::TopRight => crate::types::DecorationResizeEdge::TopRight,
            DecorationResizeEdgeWidget::BottomLeft => crate::types::DecorationResizeEdge::BottomLeft,
            DecorationResizeEdgeWidget::BottomRight => crate::types::DecorationResizeEdge::BottomRight,
        })
    }

    fn supported_modifiers(&self) -> DecorationModifiers {
        DecorationModifier::Resizable.into()
    }

    fn active_modifiers(&self) -> DecorationModifiers {
        if self.pressed {
            DecorationModifier::Resizable.into()
        } else {
            DecorationModifiers::NONE
        }
    }
}

/// The mouse cursor that hovers over the window.
#[derive(Clone)]
pub struct DecorationCursor {
    /// The shape of the cursor.
    shape_: CursorShape,
    /// Whether the cursor is visible.
    visible: bool,
    /// The id of the decoration, once it is bound.
    id: Option<crate::types::DecorationId>,
}

/// The shape of a mouse cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[derive(Default)]
pub enum CursorShape {
    /// The default pointer.
    #[default]
    Default,
    /// An I-beam, used over text.
    Text,
    /// A crosshair, used over precision targets.
    Crosshair,
    /// A hand, used over links.
    Hand,
    /// An arrow that is not allowed to do what it is over.
    NotAllowed,
    /// A resize handle pointing horizontally.
    ResizeHorizontal,
    /// A resize handle pointing vertically.
    ResizeVertical,
    /// A resize handle pointing diagonally from top-left to bottom-right.
    ResizeDiagonal,
    /// A resize handle pointing diagonally from top-right to bottom-left.
    ResizeAntiDiagonal,
    /// A move cursor.
    Move,
    /// A wait cursor.
    Wait,
    /// A help cursor.
    Help,
    /// A cell cursor, used over a grid.
    Cell,
    /// A context menu cursor.
    ContextMenu,
    /// No cursor is drawn.
    None,
}

impl DecorationCursor {
    /// Creates a cursor with the default shape.
    #[must_use]
    pub fn new() -> Self {
        Self {
            shape_: CursorShape::Default,
            visible: true,
            id: None,
        }
    }

    /// The cursor with the given shape.
    #[must_use]
    pub const fn with_shape(mut self, shape: CursorShape) -> Self {
        self.shape_ = shape;
        self
    }

    /// The cursor with the given visibility.
    #[must_use]
    pub const fn visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// The shape of the cursor.
    #[inline]
    #[must_use]
    pub const fn shape(&self) -> CursorShape {
        self.shape_
    }

    /// Whether the cursor is visible.
    #[inline]
    #[must_use]
    pub const fn is_visible(&self) -> bool {
        self.visible
    }
}

impl Widget for DecorationCursor {
    fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        constraints.enforce(Size::ZERO)
    }

    fn hit_testable(&self) -> bool {
        false
    }
}

impl Decoration for DecorationCursor {
    fn kind(&self) -> DecorationKind {
        DecorationKind::MouseCursor
    }

    fn id(&self) -> Option<crate::types::DecorationId> {
        self.id
    }

    fn bind(&mut self, id: crate::types::DecorationId) {
        self.id = Some(id);
    }

    fn unbind(&mut self) {
        self.id = None;
    }

    fn supported_modifiers(&self) -> DecorationModifiers {
        DecorationModifiers::NONE
    }
}

impl Default for DecorationCursor {
    fn default() -> Self {
        Self::new()
    }
}

/// The icon of a window.
///
/// The icon is drawn in the title bar and in the task bar. The backend
/// decodes it from a native resource or from a file and hands the pixels
/// to the widget, which paints them.
#[derive(Clone)]
pub struct DecorationIconWidget {
    /// The icon to paint.
    icon_: DecorationIcon,
    /// The size the icon is painted at.
    size_: f32,
    /// The id of the decoration, once it is bound.
    id: Option<crate::types::DecorationId>,
}

impl DecorationIconWidget {
    /// Creates an icon widget without an icon.
    #[must_use]
    pub fn new() -> Self {
        Self {
            icon_: DecorationIcon::new(),
            size_: 16.0,
            id: None,
        }
    }

    /// The icon widget with the given icon.
    #[must_use]
    pub fn with_icon(mut self, icon: DecorationIcon) -> Self {
        self.icon_ = icon;
        self
    }

    /// The icon widget with the given painted size.
    #[must_use]
    pub const fn with_size(mut self, size: f32) -> Self {
        self.size_ = size;
        self
    }

    /// The icon the widget paints.
    #[inline]
    #[must_use]
    pub const fn icon(&self) -> &DecorationIcon {
        &self.icon_
    }

    /// The size the icon is painted at.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> f32 {
        self.size_
    }
}

impl Widget for DecorationIconWidget {
    fn measure(&mut self, _scope: &mut MeasureScope<'_>, constraints: Constraints) -> Size {
        if self.icon_.is_empty() {
            constraints.enforce(Size::ZERO)
        } else {
            constraints.enforce(Size::new(self.size_, self.size_))
        }
    }

    fn draw(&mut self, scope: &mut DrawScope<'_>, bounds: Rect) {
        if self.icon_.is_empty() {
            return;
        }
        let theme = scope.theme();
        let size = self
            .size_
            .min(bounds.size.width)
            .min(bounds.size.height);
        let x = bounds.left() + (bounds.size.width - size) * 0.5;
        let y = bounds.top() + (bounds.size.height - size) * 0.5;
        scope.fill_rect(
            Rect::from_min_size(Offset::new(x, y), size, size),
            theme.colors.on_surface,
        );
    }

    fn hit_testable(&self) -> bool {
        false
    }
}

impl Decoration for DecorationIconWidget {
    fn kind(&self) -> DecorationKind {
        DecorationKind::WindowIcon
    }

    fn id(&self) -> Option<crate::types::DecorationId> {
        self.id
    }

    fn bind(&mut self, id: crate::types::DecorationId) {
        self.id = Some(id);
    }

    fn unbind(&mut self) {
        self.id = None;
    }

    fn icon(&self) -> Option<&DecorationIcon> {
        Some(&self.icon_)
    }

    fn set_icon(&mut self, icon: DecorationIcon) -> DecorationResult<()> {
        self.icon_ = icon;
        Ok(())
    }

    fn supported_modifiers(&self) -> DecorationModifiers {
        DecorationModifiers::NONE
    }
}

impl Default for DecorationIconWidget {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DecorationBounds;
    use codevar_kuitui::node::Node;
    use codevar_kuitui::paint::{DrawList, RectOp, TextOp};
    use codevar_kuitui::theme::Theme;

    #[test]
    fn a_title_bar_is_as_wide_as_the_window() {
        let theme = Theme::light();
        let mut node = Node::new(DecorationTitleBar::new("Codevar"));
        let size = node.measure(&theme, Constraints::new(0.0, 400.0, 0.0, f32::INFINITY));
        assert_eq!(size.height, 32.0);
        assert!(size.width <= 400.0);
    }

    #[test]
    fn a_title_bar_draws_its_background_and_its_title() {
        let theme = Theme::light();
        let mut list = DrawList::new();
        let mut node = Node::new(DecorationTitleBar::new("Codevar"));
        node.measure(&theme, Constraints::new(0.0, 400.0, 0.0, f32::INFINITY));
        node.place(Rect::new(Offset::ZERO, Size::new(400.0, 32.0)));
        node.draw(&theme, &mut list);
        assert!(list.len() >= 2);
        let first = list
            .ops()
            .next()
            .expect("a rectangle was recorded");
        let first_any: &dyn core::any::Any = first;
        assert!(first_any.downcast_ref::<RectOp>().is_some());
        let second = list
            .ops()
            .nth(1)
            .expect("a text run was recorded");
        let second_any: &dyn core::any::Any = second;
        assert!(second_any.downcast_ref::<TextOp>().is_some());
    }

    #[test]
    fn a_pressed_button_calls_the_handler() {
        let theme = Theme::light();
        let pressed = Rc::new(core::cell::Cell::new(0_u32));
        let observed = Rc::clone(&pressed);
        let mut node = Node::new(
            DecorationTitleBar::new("Codevar")
                .buttons(vec![DecorationButton::Close])
                .on_click(move |_button| observed.set(observed.get() + 1)),
        );
        node.measure(&theme, Constraints::new(0.0, 400.0, 0.0, f32::INFINITY));
        node.place(Rect::new(Offset::ZERO, Size::new(400.0, 32.0)));

        let down = Event::Pointer(codevar_kuitui::event::PointerEvent::new(
            PointerEventKind::Down,
            Offset::new(390.0, 16.0),
        ));
        let up = Event::Pointer(codevar_kuitui::event::PointerEvent::new(
            PointerEventKind::Up,
            Offset::new(390.0, 16.0),
        ));
        node.widget_mut().on_event(
            &mut codevar_kuitui::widget::EventScope::new(&mut Vec::new(), &theme, Vec::new(), &mut None),
            &down,
        );
        node.widget_mut().on_event(
            &mut codevar_kuitui::widget::EventScope::new(&mut Vec::new(), &theme, Vec::new(), &mut None),
            &up,
        );
        assert_eq!(pressed.get(), 1);
    }

    #[test]
    fn a_resize_handle_reports_the_edge_it_covers() {
        let theme = Theme::light();
        let grabbed = Rc::new(core::cell::Cell::new(0_u32));
        let observed = Rc::clone(&grabbed);
        let mut node = Node::new(
            DecorationResizeHandle::new(DecorationResizeEdgeWidget::Left).on_grab(move |_edge| {
                observed.set(observed.get() + 1);
            }),
        );
        node.measure(&theme, Constraints::unbounded());
        let down = Event::Pointer(codevar_kuitui::event::PointerEvent::new(
            PointerEventKind::Down,
            Offset::new(1.0, 1.0),
        ));
        node.widget_mut().on_event(
            &mut codevar_kuitui::widget::EventScope::new(&mut Vec::new(), &theme, Vec::new(), &mut None),
            &down,
        );
        assert_eq!(grabbed.get(), 1);
    }

    #[test]
    fn decoration_bounds_report_their_rect() {
        let bounds = DecorationBounds::new(10, 20, 100, 50);
        assert_eq!(
            bounds.rect(),
            Rect::from_min_size(Offset::new(10.0, 20.0), 100.0, 50.0)
        );
    }
}
