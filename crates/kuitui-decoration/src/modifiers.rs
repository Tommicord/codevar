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

//! The window decoration modifiers.
//!
//! A modifier is one aspect of the decoration state of a window: whether
//! it is maximized, fullscreen, resizable, activated, suspended, tiled
//! or closeable. Wayland expresses them through the `xdg_toplevel` state
//! and `wm_capabilities` events; Win32 expresses them through window
//! styles, extended styles and DWM window attributes.
//!
//! Every modifier that Wayland and Win32 expose is represented here, so a
//! decoration can query exactly which modifiers are active and which ones
//! it is allowed to change.

use alloc::fmt;

/// A single window decoration modifier.
///
/// The set is the union of what Wayland (`xdg_toplevel`) and Win32
/// (window styles, extended styles, DWM attributes) can express. A
/// backend maps its native representation to and from this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum DecorationModifier {
    /// The window is activated (has the keyboard focus).
    ///
    /// Wayland: the toplevel is listed in `xdg_toplevel.wm_capabilities`
    /// or the surface received an enter event from a focused seat.
    /// Win32: the window is the foreground window.
    Activated,
    /// The window is maximized.
    ///
    /// Wayland: `xdg_toplevel.set_maximized` was applied and acknowledged.
    /// Win32: the window is in the `SW_MAXIMIZE` show state or has
    /// `WS_MAXIMIZE` set.
    Maximized,
    /// The window is fullscreen.
    ///
    /// Wayland: `xdg_toplevel.set_fullscreen` was applied.
    /// Win32: the window has `WS_POPUP` or is shown maximized over the
    /// whole monitor.
    Fullscreen,
    /// The window is minimized.
    ///
    /// Wayland: `xdg_toplevel.set_minimized` was applied.
    /// Win32: the window is in the `SW_MINIMIZE` show state.
    Minimized,
    /// The window can be resized by the user.
    ///
    /// Wayland: the `xdg_toplevel.resize` edges are non-empty.
    /// Win32: the window has `WS_THICKFRAME` (part of `WS_OVERLAPPEDWINDOW`).
    Resizable,
    /// The window is tiled by the compositor or the shell.
    ///
    /// Wayland: `xdg_toplevel.configure_bounds` or `wm_capabilities`
    /// report a tiling layout.
    /// Win32: the window was placed by `SWP_*` tiling or Snap Assist.
    Tiled,
    /// The window is suspended (the compositor withheld its surface).
    ///
    /// Wayland: the last `xdg_toplevel.configure` carried serial `0`.
    /// Win32: the window has `WS_DISABLED`.
    Suspended,
    /// The window has a close button.
    ///
    /// Wayland: `xdg_toplevel.close` exists.
    /// Win32: the window has `WS_SYSMENU`.
    Closable,
    /// The window shows a window menu.
    ///
    /// Wayland: `xdg_toplevel.show_window_menu` exists.
    /// Win32: the window has `WS_SYSMENU`.
    Menu,
}

impl DecorationModifier {
    /// Whether the modifier is a state the window currently sits in,
    /// rather than a capability it advertises.
    #[must_use]
    pub const fn is_state(self) -> bool {
        matches!(
            self,
            Self::Activated | Self::Maximized | Self::Fullscreen | Self::Minimized | Self::Suspended
        )
    }

    /// Whether the modifier is a capability the window grants to the
    /// user, rather than a state it currently sits in.
    #[must_use]
    pub const fn is_capability(self) -> bool {
        !self.is_state()
    }

    /// The modifier as a single bit.
    #[inline]
    #[must_use]
    pub const fn bit(self) -> u16 {
        match self {
            DecorationModifier::Activated => 1 << 0,
            DecorationModifier::Maximized => 1 << 1,
            DecorationModifier::Fullscreen => 1 << 2,
            DecorationModifier::Minimized => 1 << 3,
            DecorationModifier::Resizable => 1 << 4,
            DecorationModifier::Tiled => 1 << 5,
            DecorationModifier::Suspended => 1 << 6,
            DecorationModifier::Closable => 1 << 7,
            DecorationModifier::Menu => 1 << 8,
        }
    }
}

impl fmt::Display for DecorationModifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Activated => "activated",
            Self::Maximized => "maximized",
            Self::Fullscreen => "fullscreen",
            Self::Minimized => "minimized",
            Self::Resizable => "resizable",
            Self::Tiled => "tiled",
            Self::Suspended => "suspended",
            Self::Closable => "closable",
            Self::Menu => "menu",
        })
    }
}

/// A set of window decoration modifiers.
///
/// Stored as a bit set so a new modifier is a new constant and not a new
/// field. Every modifier of `codevar-kuitui` maps to one bit, which is
/// what lets a decoration test whether a modifier is active with a single
/// bit operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DecorationModifiers(u16);

impl DecorationModifiers {
    /// No modifier is active.
    pub const NONE: DecorationModifiers = DecorationModifiers(0);

    /// Every modifier the type knows about.
    pub const ALL: DecorationModifiers = DecorationModifiers(0xffff);

    /// Returns `true` when every modifier of `other` is present.
    #[must_use]
    pub const fn contains(self, other: DecorationModifiers) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns `true` when `modifier` is present.
    #[must_use]
    pub const fn has(self, modifier: DecorationModifier) -> bool {
        self.0 & modifier.bit() != 0
    }

    /// Adds `modifier` to the set.
    #[must_use]
    pub const fn with(mut self, modifier: DecorationModifier) -> Self {
        self.0 |= modifier.bit();
        self
    }

    /// Removes `modifier` from the set.
    #[must_use]
    pub const fn without(mut self, modifier: DecorationModifier) -> Self {
        self.0 &= !modifier.bit();
        self
    }

    /// Toggles `modifier` in the set.
    #[must_use]
    pub const fn toggle(mut self, modifier: DecorationModifier) -> Self {
        self.0 ^= modifier.bit();
        self
    }

    /// The number of modifiers present in the set.
    #[must_use]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// The raw bits of the set.
    #[must_use]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Creates a set from raw bits.
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Iterates over the modifiers present in the set.
    #[must_use]
    pub const fn iter(self) -> DecorationModifierIter {
        DecorationModifierIter { modifiers: self, index: 0 }
    }
}

impl From<DecorationModifier> for DecorationModifiers {
    fn from(modifier: DecorationModifier) -> Self {
        Self(modifier.bit())
    }
}

impl core::ops::BitOr for DecorationModifier {
    type Output = DecorationModifiers;

    fn bitor(self, rhs: Self) -> Self::Output {
        DecorationModifiers::from(self).with(rhs)
    }
}

impl core::ops::BitOr for DecorationModifiers {
    type Output = DecorationModifiers;

    fn bitor(mut self, rhs: Self) -> Self::Output {
        self.0 |= rhs.0;
        self
    }
}

impl core::ops::BitOrAssign for DecorationModifiers {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl core::ops::BitAnd for DecorationModifiers {
    type Output = DecorationModifiers;

    fn bitand(mut self, rhs: Self) -> Self::Output {
        self.0 &= rhs.0;
        self
    }
}

impl fmt::Display for DecorationModifiers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for modifier in self.iter() {
            if !first {
                f.write_str(" | ")?;
            }
            first = false;
            write!(f, "{modifier}")?;
        }
        if first {
            f.write_str("none")
        } else {
            Ok(())
        }
    }
}

/// An iterator over the modifiers present in a set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecorationModifierIter {
    modifiers: DecorationModifiers,
    index: u16,
}

impl Iterator for DecorationModifierIter {
    type Item = DecorationModifier;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < 9 {
            let index = self.index;
            self.index += 1;
            let bit = 1u16 << index;
            if self.modifiers.0 & bit != 0 {
                return Some(match index {
                    0 => DecorationModifier::Activated,
                    1 => DecorationModifier::Maximized,
                    2 => DecorationModifier::Fullscreen,
                    3 => DecorationModifier::Minimized,
                    4 => DecorationModifier::Resizable,
                    5 => DecorationModifier::Tiled,
                    6 => DecorationModifier::Suspended,
                    7 => DecorationModifier::Closable,
                    8 => DecorationModifier::Menu,
                    _ => unreachable!(),
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modifiers_are_a_bit_set() {
        let mut set = DecorationModifiers::NONE;
        assert_eq!(set.count(), 0);
        set = set.with(DecorationModifier::Maximized);
        set = set.with(DecorationModifier::Fullscreen);
        assert!(set.has(DecorationModifier::Maximized));
        assert!(set.has(DecorationModifier::Fullscreen));
        assert!(!set.has(DecorationModifier::Minimized));
        assert_eq!(set.count(), 2);
        set = set.without(DecorationModifier::Maximized);
        assert!(!set.has(DecorationModifier::Maximized));
        assert!(set.has(DecorationModifier::Fullscreen));
        set = set.toggle(DecorationModifier::Minimized);
        assert!(set.has(DecorationModifier::Minimized));
        set = set.toggle(DecorationModifier::Minimized);
        assert!(!set.has(DecorationModifier::Minimized));
    }

    #[test]
    fn contains_reports_every_modifier() {
        let set = DecorationModifiers::from(DecorationModifier::Maximized)
            | DecorationModifiers::from(DecorationModifier::Fullscreen);
        assert!(set.contains(DecorationModifier::Maximized.into()));
        assert!(!set.contains(DecorationModifier::Minimized.into()));
    }

    #[test]
    fn iter_yields_each_modifier_once() {
        let set = DecorationModifiers::from(DecorationModifier::Activated)
            | DecorationModifiers::from(DecorationModifier::Minimized)
            | DecorationModifiers::from(DecorationModifier::Menu);
        let mut found = DecorationModifiers::NONE;
        for modifier in set.iter() {
            found = found.with(modifier);
        }
        assert_eq!(found, set);
    }

    #[test]
    fn every_modifier_is_a_state_or_a_capability() {
        for modifier in [
            DecorationModifier::Activated,
            DecorationModifier::Maximized,
            DecorationModifier::Fullscreen,
            DecorationModifier::Minimized,
            DecorationModifier::Resizable,
            DecorationModifier::Tiled,
            DecorationModifier::Suspended,
            DecorationModifier::Closable,
            DecorationModifier::Menu,
        ] {
            assert!(modifier.is_state() ^ modifier.is_capability());
        }
    }
}
