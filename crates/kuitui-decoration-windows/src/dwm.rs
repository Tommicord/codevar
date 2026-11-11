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

//! Desktop Window Manager (DWM) attributes.
//!
//! Reads and writes `DWMWA_*` window attributes through
//! `DwmSetWindowAttribute` and `DwmGetWindowAttribute`: composition,
//! dark theme, borderless (no chrome) and the cloaking state.

use codevar_logger::log_warn;

/// A DWM window attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DwmAttribute {
    /// `DWMWA_NCRENDERING_ENABLED`: enables/disables DWM composition.
    NonClientRendering,
    /// `DWMWA_CLOAKED`: cloaks the window (hides it from the user).
    Cloaked,
    /// `DWMWA_USE_IMMERSIVE_DARK_MODE`: uses the dark theme.
    ImmersiveDarkMode,
    /// `DWMWA_WINDOW_CORNER_PREFERENCE`: sets the corner style.
    CornerPreference,
    /// `DWMWA_BORDER_COLOR`: sets the border color.
    BorderColor,
    /// `DWMWA_CAPTION_COLOR`: sets the caption color.
    CaptionColor,
}

impl DwmAttribute {
    /// The `DWMWA_*` value of the attribute.
    #[cfg(windows)]
    #[must_use]
    pub const fn value(self) -> u32 {
        use windows::Win32::Graphics::Dwm::*;
        match self {
            DwmAttribute::NonClientRendering => DWMWA_NCRENDERING_ENABLED.0,
            DwmAttribute::Cloaked => DWMWA_CLOAKED.0,
            DwmAttribute::ImmersiveDarkMode => DWMWA_USE_IMMERSIVE_DARK_MODE.0,
            DwmAttribute::CornerPreference => DWMWA_WINDOW_CORNER_PREFERENCE.0,
            DwmAttribute::BorderColor => DWMWA_BORDER_COLOR.0,
            DwmAttribute::CaptionColor => DWMWA_CAPTION_COLOR.0,
        }
    }
}

/// Sets a `BOOL`-typed (u32) DWM attribute on a window.
#[cfg(windows)]
pub fn set_bool_attribute(hwnd: isize, attribute: DwmAttribute, enabled: bool) -> bool {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::Graphics::Dwm::{DWMWINDOWATTRIBUTE, DwmSetWindowAttribute};

    let value: u32 = if enabled { 1 } else { 0 };
    unsafe {
        let result = DwmSetWindowAttribute(
            HWND(hwnd as *mut _),
            DWMWINDOWATTRIBUTE(attribute.value()),
            &value as *const u32 as *const _,
            core::mem::size_of::<u32>(),
        );
        if result.is_err() {
            log_warn!("DwmSetWindowAttribute({:?}, {enabled}) failed", attribute);
            false
        } else {
            true
        }
    }
}

#[cfg(not(windows))]
pub fn set_bool_attribute(_hwnd: isize, _attribute: DwmAttribute, _enabled: bool) -> bool {
    false
}

/// Reads a `BOOL`-typed (u32) DWM attribute from a window.
#[cfg(windows)]
pub fn get_bool_attribute(hwnd: isize, attribute: DwmAttribute) -> Option<bool> {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::Graphics::Dwm::{DWMWINDOWATTRIBUTE, DwmGetWindowAttribute};

    let mut value: u32 = 0;
    unsafe {
        let result = DwmGetWindowAttribute(
            HWND(hwnd as *mut _),
            DWMWINDOWATTRIBUTE(attribute.value()),
            &mut value as *mut u32 as *mut _,
            core::mem::size_of::<u32>(),
        );
        if result.is_err() { None } else { Some(value != 0) }
    }
}

#[cfg(not(windows))]
pub fn get_bool_attribute(_hwnd: isize, _attribute: DwmAttribute) -> Option<bool> {
    None
}

/// Applies the decoration modifiers that DWM expresses to a window.
#[cfg(windows)]
pub fn apply_modifiers(hwnd: isize, modifiers: codevar_kuitui_decoration::DecorationModifiers) {
    use codevar_kuitui_decoration::DecorationModifier;

    if modifiers.has(DecorationModifier::Suspended) {
        let _ = set_bool_attribute(hwnd, DwmAttribute::Cloaked, true);
    } else {
        let _ = set_bool_attribute(hwnd, DwmAttribute::Cloaked, false);
    }
}

#[cfg(not(windows))]
pub fn apply_modifiers(_hwnd: isize, _modifiers: codevar_kuitui_decoration::DecorationModifiers) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_attribute_has_a_value() {
        for attribute in [
            DwmAttribute::NonClientRendering,
            DwmAttribute::Cloaked,
            DwmAttribute::ImmersiveDarkMode,
            DwmAttribute::CornerPreference,
            DwmAttribute::BorderColor,
            DwmAttribute::CaptionColor,
        ] {
            let _ = attribute.value();
        }
    }
}
