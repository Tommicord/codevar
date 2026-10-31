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

//! The `cl_properties` lists that describe an object at creation time.
//!
//! `clCreateBufferWithProperties` and `clCreateSemaphoreWithPropertiesKHR`
//! take one argument that the rest of this crate never had to deal with: a
//! flat array of [`sys::Properties`] words in which every property is a
//! `(name, value)` pair and the whole array ends with a single zero. Both
//! the name and the value are that same `u64`, so a file descriptor, a
//! `HANDLE`, a device pointer and a wide string pointer all occupy the
//! value slot widened exactly the way the C headers widen them.
//!
//! That widening is the reason this module exists. Casting at each call
//! site invites a truncation (`*fd as usize` is 32 bits wide on the 32-bit
//! targets this crate is built for, while the slot is always 64) and hides
//! the sign extension a negative descriptor needs, so the conversions live
//! here instead: [`PropertyList`] assembles the words and
//! [`from_int`] / [`from_ptr`] produce a slot from a Rust value.
//!
//! A built list is an ordinary `Vec`, which the caller keeps alive for the
//! duration of the driver call — the driver reads the list during the call
//! and does not retain the pointer.

use alloc::vec::Vec;

use crate::sys;

/// A `cl_properties` list under construction.
// The import calls that build a list land with the external-memory and
// semaphore modules; until then the builder is read only by the tests.
#[allow(dead_code)]
pub(crate) struct PropertyList {
    words: Vec<sys::Properties>,
}

#[allow(dead_code)]
impl PropertyList {
    /// Starts an empty list.
    pub(crate) const fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// Appends one `(name, value)` pair.
    pub(crate) fn pair(&mut self, name: sys::Properties, value: sys::Properties) {
        self.words.push(name);
        self.words.push(value);
    }

    /// Ends the list with the zero terminator the driver reads.
    pub(crate) fn build(mut self) -> Vec<sys::Properties> {
        self.words.push(0);
        self.words
    }
}

/// The value slot for an `int` property, such as a file descriptor.
///
/// The widening sign-extends, which is what the C headers do for a
/// descriptor: a driver that receives `-1` in a `cl_properties` word
/// casts it back to `int` and reads `-1` again.
// Used by the external-memory import path, which lands next.
#[allow(dead_code)]
pub(crate) const fn from_int(value: i32) -> sys::Properties {
    value as sys::Properties
}

/// The value slot for a pointer property, such as a `HANDLE`, a device
/// handle or a wide string.
#[allow(dead_code)]
pub(crate) fn from_ptr<T>(ptr: *const T) -> sys::Properties {
    ptr as usize as sys::Properties
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ffi::c_void;

    /// The driver reads names and values as one flat stream ending at the
    /// first zero, so the built vector must be exactly that.
    #[test]
    fn build_terminates_the_pairs() {
        let mut list = PropertyList::new();
        list.pair(0x2060, 7);
        list.pair(0x2051, 9);
        assert_eq!(list.build(), [0x2060, 7, 0x2051, 9, 0]);
    }

    /// An empty list is still a list the driver can read: it terminates
    /// immediately, which the API documents as "no properties".
    #[test]
    fn an_empty_list_is_a_bare_terminator() {
        assert_eq!(PropertyList::new().build(), [0]);
    }

    /// A negative descriptor must keep its value through the widening:
    /// truncating it to 32 bits or masking off the sign both make the
    /// driver import a different file.
    #[test]
    fn from_int_widens_with_the_sign_of_the_c_cast() {
        assert_eq!(from_int(-1), u64::MAX);
        assert_eq!(from_int(i32::MIN), 0xFFFF_FFFF_8000_0000);
        assert_eq!(from_int(42), 42);
    }

    /// A pointer property must survive the widening unchanged, including on
    /// targets where a pointer is narrower than the slot.
    #[test]
    fn from_ptr_keeps_the_address_of_the_pointee() {
        let value: u16 = 1;
        let slot = from_ptr(&value);
        assert_eq!(slot as usize, &value as *const u16 as usize);
        assert_eq!(from_ptr(core::ptr::null::<c_void>()), 0);
    }
}
