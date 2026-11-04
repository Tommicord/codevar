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

//! The options, the colors and the shadow a button is drawn with.
//!
//! A button answers three questions at draw time: which flags the
//! application gave it ([`ButtonFlags`]), which interaction states the
//! pointer put it in (`InteractionState`, owned by the framework) and
//! which colors and shadow follow from that pair
//! ([`ButtonColors::container_for`], [`ButtonColors::overlay_for`],
//! [`ButtonElevation::sigma_for`]). Resolving them at draw time is what
//! lets a press repaint without recomposing the tree.

use core::ops::{BitOr, BitOrAssign};

use codevar_kuitui::{Color, ColorScheme, CornerRadii, InteractionState};

use crate::defaults;

/// What a button does, as a set of options instead of booleans.
///
/// The flags combine: a button is enabled *and* tracked for hover *and*
/// takes the focus, so one word carries all three, and an option a
/// button does not have costs nothing at draw time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ButtonFlags(u8);

impl ButtonFlags {
    /// The button only paints itself.
    pub const NONE: ButtonFlags = ButtonFlags(0);
    /// The button answers a click. Clearing this bit is how a button is
    /// disabled: the press is never reported and the content color is
    /// the disabled one.
    pub const ENABLED: ButtonFlags = ButtonFlags(1 << 0);
    /// The button reports the pointer to its interaction source, which
    /// is what makes the hover and press colors appear.
    pub const HOVERABLE: ButtonFlags = ButtonFlags(1 << 1);
    /// The button takes the keyboard focus when a button is pressed
    /// inside it, and reports [`InteractionState::FOCUSED`] to its
    /// source.
    pub const FOCUSABLE: ButtonFlags = ButtonFlags(1 << 2);
    /// The button flips [`InteractionState::SELECTED`] when a button is
    /// released inside it, the way a toggle button does.
    pub const SELECTABLE: ButtonFlags = ButtonFlags(1 << 3);
    /// An enabled button that tracks the pointer: what Compose's
    /// `Button` draws when the caller leaves every parameter defaulted.
    pub const STANDARD: ButtonFlags = ButtonFlags(Self::ENABLED.0 | Self::HOVERABLE.0);
    /// A button that does everything the framework can ask of it.
    pub const ALL: ButtonFlags =
        ButtonFlags(Self::ENABLED.0 | Self::HOVERABLE.0 | Self::FOCUSABLE.0 | Self::SELECTABLE.0);

    /// The underlying bits.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every bit of `other` is set.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: ButtonFlags) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no option at all is selected.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The set with the bits of `other` added.
    #[inline]
    #[must_use]
    pub const fn union(self, other: ButtonFlags) -> Self {
        ButtonFlags(self.0 | other.0)
    }

    /// The set with the bits of `other` removed.
    #[inline]
    #[must_use]
    pub const fn without(self, other: ButtonFlags) -> Self {
        ButtonFlags(self.0 & !other.0)
    }
}

impl BitOr for ButtonFlags {
    type Output = ButtonFlags;

    #[inline]
    fn bitor(self, rhs: ButtonFlags) -> ButtonFlags {
        ButtonFlags(self.0 | rhs.0)
    }
}

impl BitOrAssign for ButtonFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: ButtonFlags) {
        self.0 |= rhs.0;
    }
}

/// The colors of the container, the state layer and the content of a
/// button.
///
/// The container is the flat color of the button and the state layer is
/// the translucent tint the Material 3 state layers put over it, so the
/// two are drawn as two rectangles instead of being blended here: the
/// backend composites the layer, which is what keeps a button with a
/// transparent container — a text button — tinting correctly on hover.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ButtonColors {
    /// The container at rest.
    pub container: Color,
    /// The content — the label and the icons — at rest.
    pub content: Color,
    /// The container while the button is selected.
    pub selected_container: Color,
    /// The content while the button is selected.
    pub selected_content: Color,
    /// The container when [`ButtonFlags::ENABLED`] is clear.
    pub disabled_container: Color,
    /// The content when [`ButtonFlags::ENABLED`] is clear.
    pub disabled_content: Color,
    /// The color the state layer is tinted with.
    pub state_layer: Color,
}

impl ButtonColors {
    /// The colors of a button whose states tint `container` with
    /// `state_layer` and whose content is `content`.
    ///
    /// The disabled colors follow Material 3: the container becomes the
    /// state layer at 12 percent and the content keeps its own color at
    /// 38 percent alpha. A button that selects keeps both colors until
    /// [`ButtonColors::with_selected`] or [`ButtonColors::with_selected_content`]
    /// says otherwise, because a role without a selection has no
    /// selected state to draw.
    #[inline]
    #[must_use]
    pub const fn layered(container: Color, content: Color, state_layer: Color) -> Self {
        Self {
            container,
            content,
            selected_container: container,
            selected_content: content,
            disabled_container: state_layer.with_alpha(defaults::DISABLED_CONTAINER_ALPHA),
            disabled_content: content.with_alpha(defaults::DISABLED_CONTENT_ALPHA),
            state_layer,
        }
    }

    /// The colors with the container of a selected button replaced.
    ///
    /// The content keeps its color, which is only legible while the two
    /// still contrast; [`ButtonColors::with_selected_content`] is the
    /// variant for a selected container that needs its own `on_*` role.
    #[inline]
    #[must_use]
    pub const fn with_selected(mut self, container: Color) -> Self {
        self.selected_container = container;
        self
    }

    /// The colors with both the container and the content of a selected
    /// button replaced.
    ///
    /// A selection is not a transient state — Material 3 swaps the
    /// container for a different role color, and the content has to
    /// follow it into the `on_*` color of that role or the label of the
    /// selected button is drawn in the color of the container it no
    /// longer has.
    #[inline]
    #[must_use]
    pub const fn with_selected_content(mut self, container: Color, content: Color) -> Self {
        self.selected_container = container;
        self.selected_content = content;
        self
    }

    /// The filled button of `scheme`: the primary container, drawn with
    /// no border and an `on_surface` state layer.
    #[inline]
    #[must_use]
    pub const fn filled(scheme: &ColorScheme) -> Self {
        Self::layered(scheme.primary, scheme.on_primary, scheme.on_surface)
            .with_selected_content(scheme.primary_container, scheme.on_primary_container)
    }

    /// The tonal button of `scheme`: the primary container role.
    #[inline]
    #[must_use]
    pub const fn tonal(scheme: &ColorScheme) -> Self {
        Self::layered(
            scheme.primary_container,
            scheme.on_primary_container,
            scheme.on_surface,
        )
        .with_selected_content(scheme.secondary, scheme.on_secondary)
    }

    /// The outlined button of `scheme`: no container at all, so only the
    /// state layer and [`BorderStroke::outline`] are visible.
    #[inline]
    #[must_use]
    pub const fn outlined(scheme: &ColorScheme) -> Self {
        Self::layered(Color::TRANSPARENT, scheme.on_surface, scheme.on_surface)
            .with_selected_content(scheme.surface_variant, scheme.on_surface_variant)
    }

    /// The text button of `scheme`: no container, the primary color as
    /// content.
    #[inline]
    #[must_use]
    pub const fn text(scheme: &ColorScheme) -> Self {
        Self::layered(Color::TRANSPARENT, scheme.primary, scheme.on_surface)
    }

    /// The container color `state` and `flags` ask for.
    ///
    /// A disabled button wins over every interaction, then the choice
    /// decides the container; the transient states are the overlay, not
    /// the container, so they never reach this function.
    #[inline]
    #[must_use]
    pub fn container_for(&self, state: InteractionState, flags: ButtonFlags) -> Color {
        if !flags.contains(ButtonFlags::ENABLED) {
            return self.disabled_container;
        }
        if flags.contains(ButtonFlags::SELECTABLE) && state.contains(InteractionState::SELECTED) {
            return self.selected_container;
        }
        self.container
    }

    /// The state layer drawn over the container, if any.
    ///
    /// The highest-priority state is the only one applied: a button that
    /// is pressed and hovered does not darken twice, which is what
    /// Material 3 prescribes and what keeps a press legible.
    #[must_use]
    pub fn overlay_for(&self, state: InteractionState, flags: ButtonFlags) -> Color {
        if !flags.contains(ButtonFlags::HOVERABLE) || !flags.contains(ButtonFlags::ENABLED) {
            return Color::TRANSPARENT;
        }
        let alpha = if state.contains(InteractionState::PRESSED) {
            defaults::PRESSED_OVERLAY_ALPHA
        } else if state.contains(InteractionState::FOCUSED) {
            defaults::FOCUSED_OVERLAY_ALPHA
        } else if state.contains(InteractionState::HOVERED) {
            defaults::HOVER_OVERLAY_ALPHA
        } else {
            return Color::TRANSPARENT;
        };
        self.state_layer.with_alpha(alpha)
    }

    /// The content color `state` and `flags` ask for.
    ///
    /// Only the selection changes the content, because only it changes
    /// the container under the content; the transient states keep the
    /// content color and tint the container, which is what Material 3
    /// prescribes and what keeps a hover from flickering the label.
    #[inline]
    #[must_use]
    pub fn content_for(&self, state: InteractionState, flags: ButtonFlags) -> Color {
        if !flags.contains(ButtonFlags::ENABLED) {
            return self.disabled_content;
        }
        if flags.contains(ButtonFlags::SELECTABLE) && state.contains(InteractionState::SELECTED) {
            return self.selected_content;
        }
        self.content
    }
}

/// The colors a button draws with, or the role that picks them from the
/// theme.
///
/// Compose defaults a button's colors by reading the color scheme from
/// the composition, and Kuitui only has a theme at measure and draw
/// time. A `ButtonPalette` keeps both possible: an application that
/// names the colors gets them, an application that names a role gets
/// the theme's colors for it.
///
/// Two palettes are not compared: the role is a function, and Rust does
/// not promise that the same function has the same address in every
/// codegen unit. Compare the `colors`, or resolve the palette against a
/// scheme and compare the result.
#[derive(Debug, Clone, Copy)]
pub struct ButtonPalette {
    /// The colors the application asked for, if it asked for any.
    pub colors: Option<ButtonColors>,
    /// The role that answers for the theme when `colors` is `None`.
    pub role: fn(&ColorScheme) -> ButtonColors,
}

impl ButtonPalette {
    /// The palette that always draws with `colors`.
    #[inline]
    #[must_use]
    pub const fn fixed(colors: ButtonColors) -> Self {
        Self {
            colors: Some(colors),
            role: ButtonColors::filled,
        }
    }

    /// The palette that asks the theme for the colors of `role`.
    #[inline]
    #[must_use]
    pub const fn role_of(role: fn(&ColorScheme) -> ButtonColors) -> Self {
        Self { colors: None, role }
    }

    /// The colors of the palette in `scheme`.
    #[inline]
    #[must_use]
    pub fn resolve(&self, scheme: &ColorScheme) -> ButtonColors {
        match self.colors {
            Some(colors) => colors,
            None => (self.role)(scheme),
        }
    }
}

/// The ambient shadow a button draws under its container.
///
/// The three sigmas are the standard deviations of the blur the layer
/// program evaluates for the silhouette of the container, in logical
/// pixels; a shadow of `0.0` is no shadow at all, so a filled button
/// with [`ButtonElevation::flat`] costs nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ButtonElevation {
    /// The sigma while the button is at rest.
    pub resting: f32,
    /// The sigma while the pointer is inside the button.
    pub hovered: f32,
    /// The sigma while a button is held inside the button.
    pub pressed: f32,
    /// The color of the silhouette.
    pub color: Color,
}

impl ButtonElevation {
    /// The elevation with the three sigmas and the shadow `color`.
    #[inline]
    #[must_use]
    pub const fn new(resting: f32, hovered: f32, pressed: f32, color: Color) -> Self {
        Self {
            resting,
            hovered,
            pressed,
            color,
        }
    }

    /// No shadow in any state.
    #[inline]
    #[must_use]
    pub const fn flat() -> Self {
        Self {
            resting: 0.0,
            hovered: 0.0,
            pressed: 0.0,
            color: Color::TRANSPARENT,
        }
    }

    /// The elevated button: a shadow at rest that grows under the
    /// pointer and disappears while the button is held.
    #[inline]
    #[must_use]
    pub const fn elevated() -> Self {
        Self::new(
            defaults::SHADOW_RESTING,
            defaults::SHADOW_HOVERED,
            defaults::SHADOW_PRESSED,
            defaults::SHADOW_COLOR,
        )
    }

    /// The sigma of the shadow `state` and `flags` ask for, or `None`
    /// when the button draws no shadow at all.
    #[must_use]
    pub fn sigma_for(&self, state: InteractionState, flags: ButtonFlags) -> Option<f32> {
        if !flags.contains(ButtonFlags::ENABLED) || !flags.contains(ButtonFlags::HOVERABLE) {
            return None;
        }
        let sigma = if state.contains(InteractionState::PRESSED) {
            self.pressed
        } else if state.contains(InteractionState::HOVERED) {
            self.hovered
        } else {
            self.resting
        };
        (sigma > 0.0).then_some(sigma)
    }
}

/// The outline a button draws around its container.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BorderStroke {
    /// The width of the outline, in logical pixels.
    pub width: f32,
    /// The color of the outline.
    pub color: Color,
    /// The radii of the corners of the outline.
    pub radii: Option<CornerRadii>,
}

impl BorderStroke {
    /// The outline of `width` pixels and `color`, rounded with the shape
    /// of the button it decorates.
    #[inline]
    #[must_use]
    pub const fn new(width: f32, color: Color) -> Self {
        Self {
            width,
            color,
            radii: None,
        }
    }

    /// The outline with corners of its own instead of the button's.
    #[inline]
    #[must_use]
    pub const fn with_radii(mut self, radii: CornerRadii) -> Self {
        self.radii = Some(radii);
        self
    }

    /// The outline of the outlined button of `scheme`.
    #[inline]
    #[must_use]
    pub const fn outline(scheme: &ColorScheme) -> Self {
        Self::new(defaults::OUTLINE_WIDTH, scheme.outline)
    }

    /// The radii the outline is drawn with, given the shape of the
    /// button.
    #[inline]
    #[must_use]
    pub fn radii_for(&self, shape: CornerRadii) -> CornerRadii {
        self.radii.unwrap_or(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEME: ColorScheme = ColorScheme::light();

    #[test]
    fn flags_combine_with_the_or_operator() {
        let flags = ButtonFlags::ENABLED | ButtonFlags::FOCUSABLE;
        assert!(flags.contains(ButtonFlags::ENABLED));
        assert!(flags.contains(ButtonFlags::FOCUSABLE));
        assert!(!flags.contains(ButtonFlags::SELECTABLE));
        assert_eq!(flags.bits(), 0b0101);
    }

    #[test]
    fn the_standard_flag_set_is_an_enabled_hoverable_button() {
        assert!(ButtonFlags::STANDARD.contains(ButtonFlags::ENABLED | ButtonFlags::HOVERABLE));
        assert!(!ButtonFlags::STANDARD.contains(ButtonFlags::FOCUSABLE));
        assert!(ButtonFlags::ALL.contains(ButtonFlags::STANDARD));
    }

    #[test]
    fn a_disabled_button_ignores_every_state() {
        let colors = ButtonColors::filled(&SCHEME);
        let state = InteractionState::PRESSED | InteractionState::HOVERED;
        let disabled = ButtonFlags::HOVERABLE;
        assert_eq!(colors.container_for(state, disabled), colors.disabled_container);
        assert_eq!(colors.overlay_for(state, disabled), Color::TRANSPARENT);
        assert_eq!(colors.content_for(state, disabled), colors.disabled_content);
    }

    #[test]
    fn the_press_wins_over_the_hover_of_the_same_pointer() {
        let colors = ButtonColors::layered(Color::WHITE, Color::BLACK, Color::rgb(0.1, 0.1, 0.1));
        let overlay = colors.overlay_for(
            InteractionState::PRESSED | InteractionState::HOVERED,
            ButtonFlags::STANDARD,
        );
        assert_eq!(overlay.alpha, defaults::PRESSED_OVERLAY_ALPHA);
        let hovered = colors.overlay_for(InteractionState::HOVERED, ButtonFlags::STANDARD);
        assert_eq!(hovered.alpha, defaults::HOVER_OVERLAY_ALPHA);
    }

    #[test]
    fn only_a_selectable_button_shows_the_selected_container() {
        let colors = ButtonColors::filled(&SCHEME).with_selected(Color::RED);
        let selected = InteractionState::SELECTED;
        assert_eq!(
            colors.container_for(selected, ButtonFlags::STANDARD),
            ButtonColors::filled(&SCHEME).container
        );
        let toggle = ButtonFlags::STANDARD | ButtonFlags::SELECTABLE;
        assert_eq!(colors.container_for(selected, toggle), Color::RED);
    }

    #[test]
    fn a_selected_button_keeps_the_content_of_the_rest_until_told() {
        let colors = ButtonColors::layered(Color::WHITE, Color::BLACK, Color::rgb(0.5, 0.5, 0.5))
            .with_selected(Color::RED);
        let toggle = ButtonFlags::STANDARD | ButtonFlags::SELECTABLE;
        assert_eq!(
            colors.content_for(InteractionState::SELECTED, toggle),
            colors.content,
            "a container of its own does not imply a content of its own"
        );
    }

    #[test]
    fn the_selected_content_follows_the_selected_container() {
        let colors = ButtonColors::layered(Color::WHITE, Color::BLACK, Color::rgb(0.5, 0.5, 0.5))
            .with_selected_content(Color::RED, Color::BLUE);
        let toggle = ButtonFlags::STANDARD | ButtonFlags::SELECTABLE;
        assert_eq!(
            colors.container_for(InteractionState::SELECTED, toggle),
            Color::RED
        );
        assert_eq!(
            colors.content_for(InteractionState::SELECTED, toggle),
            Color::BLUE
        );
        assert_eq!(
            colors.content_for(InteractionState::NONE, toggle),
            Color::BLACK,
            "the rest of the states keep the content of the role"
        );
    }

    #[test]
    fn each_role_selects_the_content_of_the_container_it_moves_to() {
        let filled = ButtonColors::filled(&SCHEME);
        assert_eq!(filled.selected_container, SCHEME.primary_container);
        assert_eq!(filled.selected_content, SCHEME.on_primary_container);
        let tonal = ButtonColors::tonal(&SCHEME);
        assert_eq!(tonal.selected_container, SCHEME.secondary);
        assert_eq!(tonal.selected_content, SCHEME.on_secondary);
        let outlined = ButtonColors::outlined(&SCHEME);
        assert_eq!(outlined.selected_container, SCHEME.surface_variant);
        assert_eq!(outlined.selected_content, SCHEME.on_surface_variant);
    }

    #[test]
    fn a_text_button_has_no_container_to_paint() {
        let colors = ButtonColors::text(&SCHEME);
        assert_eq!(colors.container, Color::TRANSPARENT);
        assert_eq!(colors.content, SCHEME.primary);
    }

    #[test]
    fn a_hoverable_off_button_never_tints() {
        let colors = ButtonColors::filled(&SCHEME);
        let flags = ButtonFlags::ENABLED;
        assert_eq!(
            colors.overlay_for(InteractionState::HOVERED, flags),
            Color::TRANSPARENT
        );
    }

    #[test]
    fn a_flat_elevation_draws_no_shadow() {
        assert_eq!(
            ButtonElevation::flat().sigma_for(InteractionState::HOVERED, ButtonFlags::STANDARD),
            None
        );
        let elevated = ButtonElevation::elevated();
        assert_eq!(
            elevated.sigma_for(InteractionState::NONE, ButtonFlags::STANDARD),
            Some(defaults::SHADOW_RESTING)
        );
        assert_eq!(
            elevated.sigma_for(InteractionState::PRESSED, ButtonFlags::STANDARD),
            None,
            "a pressed elevated button has no shadow at all"
        );
        assert_eq!(
            elevated.sigma_for(InteractionState::HOVERED, ButtonFlags::STANDARD),
            Some(defaults::SHADOW_HOVERED)
        );
    }

    #[test]
    fn a_palette_without_colors_asks_the_theme() {
        let palette = ButtonPalette::role_of(ButtonColors::tonal);
        assert_eq!(palette.resolve(&SCHEME), ButtonColors::tonal(&SCHEME));
        let fixed = ButtonPalette::fixed(ButtonColors::text(&SCHEME));
        assert_eq!(fixed.resolve(&SCHEME), ButtonColors::text(&SCHEME));
    }

    #[test]
    fn a_border_follows_the_shape_of_its_button() {
        let border = BorderStroke::outline(&SCHEME);
        assert_eq!(border.radii_for(defaults::SHAPE), defaults::SHAPE);
        let square = border.with_radii(CornerRadii::ZERO);
        assert_eq!(square.radii_for(defaults::SHAPE), CornerRadii::ZERO);
    }
}
