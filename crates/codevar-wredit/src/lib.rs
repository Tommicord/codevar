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

#![cfg_attr(not(test), no_std)]
extern crate alloc;

pub mod arch;
pub mod change;
mod codebook;
pub mod compress;
pub mod cursor_allocator;
pub mod dispatch;
pub mod edit;
pub mod error;
pub mod hot;
pub mod simd;
pub mod staging_gap;
pub mod storage;
