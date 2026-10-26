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

//! Microphone input: the level of a captured block and the handlers that read
//! it.
//!
//! A microphone is not a discrete device like a keyboard: it produces blocks
//! of samples several thousand times per second. Keeping every block would
//! allocate on the audio path and hand the UI more data than it can draw, so
//! the state keeps what a level meter needs — root mean square, peak,
//! clipping, crossings — and the handler receives the block itself while it is
//! still in memory.
//!
//! The measurement is deliberately independent of the capture rate: the levels
//! describe the samples that were passed in, and the helpers that need a
//! duration take the sample layout of the block. The time-based smoothing is
//! in the state, where the coefficient is derived from the rate of the block
//! being applied, so one handler behaves the same at 8, 16, 44.1 and 48 kHz.

mod level;

pub use level::AudioLevels;
