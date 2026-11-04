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

//! Device classes and the bitsets used to name a group of them.

use core::fmt;

/// A kind of input device.
///
/// The class is the unit of *registration*: a backend starts the operating
/// system resources for a class the first time a caller opens a stream for
/// it, and releases them again when the caller unregisters the class. A
/// device reports one class, so opening [`DeviceClass::Pointer`] never wakes
/// the keyboard path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum DeviceClass {
    /// A keyboard, including keymaps and text input.
    Keyboard,
    /// A mouse, trackpad, pen or touch surface that reports pointer motion.
    Pointer,
    /// A microphone or another audio capture device.
    Microphone,
    /// A game controller.
    Gamepad,
}

impl DeviceClass {
    /// Every class this crate knows about.
    pub const ALL: [DeviceClass; 4] = [
        DeviceClass::Keyboard,
        DeviceClass::Pointer,
        DeviceClass::Microphone,
        DeviceClass::Gamepad,
    ];

    /// The name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            DeviceClass::Keyboard => "keyboard",
            DeviceClass::Pointer => "pointer",
            DeviceClass::Microphone => "microphone",
            DeviceClass::Gamepad => "gamepad",
        }
    }

    /// The single bit that represents this class in a [`ClassSet`].
    #[must_use]
    pub const fn bit(self) -> u16 {
        1 << (self as u16)
    }
}

impl fmt::Display for DeviceClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of [`DeviceClass`] values.
///
/// The set is a `u16` bitset rather than a slice so a stream carries it by
/// value and the hub tests membership with one mask operation on every
/// event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct ClassSet(u16);

impl ClassSet {
    /// No class.
    pub const NONE: ClassSet = ClassSet(0);
    /// Only the keyboard.
    pub const KEYBOARD: ClassSet = ClassSet(DeviceClass::Keyboard.bit());
    /// Only the pointer.
    pub const POINTER: ClassSet = ClassSet(DeviceClass::Pointer.bit());
    /// Only microphones.
    pub const MICROPHONE: ClassSet = ClassSet(DeviceClass::Microphone.bit());
    /// Only gamepads.
    pub const GAMEPAD: ClassSet = ClassSet(DeviceClass::Gamepad.bit());
    /// Every class.
    pub const ALL: ClassSet = ClassSet(
        DeviceClass::Keyboard.bit()
            | DeviceClass::Pointer.bit()
            | DeviceClass::Microphone.bit()
            | DeviceClass::Gamepad.bit(),
    );

    /// The set holding exactly `class`.
    #[must_use]
    pub const fn from_class(class: DeviceClass) -> Self {
        Self(class.bit())
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether `class` is in the set.
    #[must_use]
    pub const fn contains(self, class: DeviceClass) -> bool {
        (self.0 & class.bit()) != 0
    }

    /// Whether every class of `other` is in the set.
    #[must_use]
    pub const fn contains_all(self, other: ClassSet) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether the two sets share at least one class.
    #[must_use]
    pub const fn intersects(self, other: ClassSet) -> bool {
        (self.0 & other.0) != 0
    }

    /// Adds `class` to the set.
    pub fn insert(&mut self, class: DeviceClass) {
        self.0 |= class.bit();
    }

    /// Removes `class` from the set.
    pub fn remove(&mut self, class: DeviceClass) {
        self.0 &= !class.bit();
    }

    /// The raw bits, for backends that need to pass the set across an FFI
    /// boundary.
    #[must_use]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Iterates over the classes in the set in [`DeviceClass::ALL`] order.
    pub fn iter(self) -> impl Iterator<Item = DeviceClass> {
        DeviceClass::ALL
            .into_iter()
            .filter(move |class| self.contains(*class))
    }
}

impl From<DeviceClass> for ClassSet {
    fn from(class: DeviceClass) -> Self {
        ClassSet::from_class(class)
    }
}

impl FromIterator<DeviceClass> for ClassSet {
    fn from_iter<I: IntoIterator<Item = DeviceClass>>(iter: I) -> Self {
        let mut set = ClassSet::NONE;
        for class in iter {
            set.insert(class);
        }
        set
    }
}

impl core::ops::BitOr for ClassSet {
    type Output = ClassSet;

    fn bitor(self, rhs: ClassSet) -> ClassSet {
        ClassSet(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for ClassSet {
    fn bitor_assign(&mut self, rhs: ClassSet) {
        self.0 |= rhs.0;
    }
}

impl core::ops::BitAnd for ClassSet {
    type Output = ClassSet;

    fn bitand(self, rhs: ClassSet) -> ClassSet {
        ClassSet(self.0 & rhs.0)
    }
}

impl core::ops::Not for ClassSet {
    type Output = ClassSet;

    fn not(self) -> ClassSet {
        ClassSet(!self.0 & ClassSet::ALL.0)
    }
}

/// What a device can report.
///
/// Capabilities refine a class: a pointer that reports motion but has no
/// wheel has [`DeviceCapability::POINTER_MOTION`] without
/// [`DeviceCapability::SCROLL`]. The set lets a caller reject a device before
/// subscribing to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct DeviceCapability(u16);

impl DeviceCapability {
    /// No capability.
    pub const NONE: DeviceCapability = DeviceCapability(0);
    /// Reports key transitions.
    pub const KEY: DeviceCapability = DeviceCapability(1 << 0);
    /// Reports text produced by the active keymap or input method.
    pub const TEXT: DeviceCapability = DeviceCapability(1 << 1);
    /// Reports relative or absolute pointer motion.
    pub const POINTER_MOTION: DeviceCapability = DeviceCapability(1 << 2);
    /// Reports pointer button transitions.
    pub const POINTER_BUTTONS: DeviceCapability = DeviceCapability(1 << 3);
    /// Reports wheel or trackpad scrolling.
    pub const SCROLL: DeviceCapability = DeviceCapability(1 << 4);
    /// Reports audio capture frames.
    pub const AUDIO_CAPTURE: DeviceCapability = DeviceCapability(1 << 5);
    /// Reports gamepad buttons and axes.
    pub const GAMEPAD: DeviceCapability = DeviceCapability(1 << 6);

    /// The capability a class normally provides.
    #[must_use]
    pub const fn for_class(class: DeviceClass) -> Self {
        match class {
            DeviceClass::Keyboard => DeviceCapability(DeviceCapability::KEY.0 | DeviceCapability::TEXT.0),
            DeviceClass::Pointer => DeviceCapability(
                DeviceCapability::POINTER_MOTION.0
                    | DeviceCapability::POINTER_BUTTONS.0
                    | DeviceCapability::SCROLL.0,
            ),
            DeviceClass::Microphone => DeviceCapability::AUDIO_CAPTURE,
            DeviceClass::Gamepad => DeviceCapability::GAMEPAD,
        }
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every capability of `other` is present.
    #[must_use]
    pub const fn contains(self, other: DeviceCapability) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Adds `capability` to the set.
    pub fn insert(&mut self, capability: DeviceCapability) {
        self.0 |= capability.0;
    }

    /// The raw bits, for backends that keep their own capability table.
    #[must_use]
    pub const fn bits(self) -> u16 {
        self.0
    }
}

impl core::ops::BitOr for DeviceCapability {
    type Output = DeviceCapability;

    fn bitor(self, rhs: DeviceCapability) -> DeviceCapability {
        DeviceCapability(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for DeviceCapability {
    fn bitor_assign(&mut self, rhs: DeviceCapability) {
        self.0 |= rhs.0;
    }
}

#[cfg(test)]
mod tests {
    use super::{ClassSet, DeviceCapability, DeviceClass};

    #[test]
    fn class_set_insert_remove_and_intersect() {
        let mut set = ClassSet::KEYBOARD;
        set.insert(DeviceClass::Pointer);
        assert!(set.contains(DeviceClass::Keyboard));
        assert!(set.contains(DeviceClass::Pointer));
        assert!(!set.contains(DeviceClass::Microphone));
        assert!(set.intersects(ClassSet::POINTER | ClassSet::MICROPHONE));
        assert!(set.contains_all(ClassSet::KEYBOARD));

        set.remove(DeviceClass::Keyboard);
        assert!(!set.contains(DeviceClass::Keyboard));
        assert_eq!(set, ClassSet::POINTER);
    }

    #[test]
    fn class_set_iterates_in_declaration_order() {
        let classes: alloc::vec::Vec<_> = (ClassSet::MICROPHONE | ClassSet::KEYBOARD)
            .iter()
            .collect();
        assert_eq!(
            classes,
            alloc::vec![DeviceClass::Keyboard, DeviceClass::Microphone]
        );
    }

    #[test]
    fn complement_stays_inside_the_known_classes() {
        let complement = !ClassSet::KEYBOARD;
        assert!(!complement.contains(DeviceClass::Keyboard));
        assert!(complement.contains(DeviceClass::Pointer));
        assert_eq!(complement | ClassSet::KEYBOARD, ClassSet::ALL);
    }

    #[test]
    fn every_class_provides_the_capability_it_names() {
        assert!(DeviceCapability::for_class(DeviceClass::Keyboard).contains(DeviceCapability::KEY));
        assert!(DeviceCapability::for_class(DeviceClass::Keyboard).contains(DeviceCapability::TEXT));
        assert!(DeviceCapability::for_class(DeviceClass::Pointer).contains(DeviceCapability::SCROLL));
        assert!(
            DeviceCapability::for_class(DeviceClass::Microphone).contains(DeviceCapability::AUDIO_CAPTURE)
        );
        assert!(DeviceCapability::for_class(DeviceClass::Gamepad).contains(DeviceCapability::GAMEPAD));
    }
}
