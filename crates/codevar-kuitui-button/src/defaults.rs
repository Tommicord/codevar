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

//! The numbers a button is drawn with when the application asks for the
//! defaults.
//!
//! Every constant here is a Material 3 number, in logical pixels for the
//! geometry and in alpha for the state layers. They are in one file so
//! that a port to another design language replaces this module and
//! nothing else: the widgets read the numbers, they do not hard-code
//! them.

use codevar_kuitui::{Color, CornerRadii, EdgeInsets};

/// The fully-rounded shape of a Material 3 button.
///
/// The radius is half the height the vertical content padding produces,
/// which is why a button looks like a capsule and not like a rounded
/// rectangle.
pub const SHAPE: CornerRadii = CornerRadii::uniform(20.0);

/// The space between the container of a button and its content: 24
/// logical pixels on the leading and trailing edge, 12 on the top and
/// the bottom.
pub const CONTENT_PADDING: EdgeInsets = EdgeInsets::symmetric(24.0, 12.0);

/// The space between two children of the content of a button, which is
/// where an icon meets its label.
pub const CONTENT_SPACING: f32 = 8.0;

/// The width of the outline of an outlined button.
pub const OUTLINE_WIDTH: f32 = 1.0;

/// The alpha of the state layer while the pointer is inside the button.
pub const HOVER_OVERLAY_ALPHA: f32 = 0.08;

/// The alpha of the state layer while a button is held inside it.
pub const PRESSED_OVERLAY_ALPHA: f32 = 0.10;

/// The alpha of the state layer while the button has the keyboard focus.
pub const FOCUSED_OVERLAY_ALPHA: f32 = 0.12;

/// The alpha of the `on` color a disabled container is drawn with.
pub const DISABLED_CONTAINER_ALPHA: f32 = 0.12;

/// The alpha of the `on` color a disabled content is drawn with.
pub const DISABLED_CONTENT_ALPHA: f32 = 0.38;

/// The color of the silhouette a button's shadow is drawn with.
pub const SHADOW_COLOR: Color = Color::BLACK.with_alpha(0.35);

/// The blur sigma of the shadow of an elevated button at rest.
pub const SHADOW_RESTING: f32 = 1.0;

/// The blur sigma of the shadow of an elevated button under the pointer.
pub const SHADOW_HOVERED: f32 = 3.0;

/// The blur sigma of the shadow of an elevated button while it is held:
/// the shadow of a pressed button is gone, which is what makes the press
/// read as a touch rather than as a hover.
pub const SHADOW_PRESSED: f32 = 0.0;
