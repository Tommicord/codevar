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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing
//! permissions and limitations under the License.

//! Anti-aliased glyph renderer
//!
//! Faithful port of FreeType 2.6's `src/smooth` module:
//!
//! * [`grays`] — `ftgrays.c`, the exact-coverage antialiasing scan
//!   converter and its [`grays::FT_GRAYS_RASTER`] descriptor.
//! * `smooth` — `ftsmooth.c`, the renderer module that grid-fits a
//!   slot's outline, allocates the target bitmap and drives the raster
//!   (including the LCD subpixel variants).

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

extern crate alloc;

pub mod grays;
