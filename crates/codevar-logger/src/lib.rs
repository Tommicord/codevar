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

//! Codevar Logger - High-performance async logging with console utilities
//!
//! This crate provides:
//! - **Async Logging Framework**: Zero-allocation, platform-aware logging with timestamps
//! - **Console Utilities**: ANSI escape codes, cursor control, screen clearing, styles
//! - **Terminal Capabilities**: Detection and management of terminal features
//!
//! Features:
//! - Log levels: debug, info, error, irr (irrecoverable)
//! - Timestamped logging (UTC via codevar-timeutil)
//! - Raw logging (no timestamp, no level)
//! - No heap allocation (streams bytes directly)
//! - UTF-8/Unicode support via codevar-textlike-encode
//! - ANSI color/style support via consoleutil
//! - Cross-platform: Windows, Linux, macOS, FreeBSD, Android, iOS, WASM

extern crate alloc;

pub mod console;
pub mod log;

pub use console::*;
pub use log::*;
