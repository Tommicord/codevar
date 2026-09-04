//! Copyright 2026 Codevar
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(unsafe_code)]

extern crate alloc;
extern crate core;

#[cfg(target_os = "android")]
pub mod android;
pub mod base;
pub mod edit;
pub mod fcware;
pub mod logtrace;
pub mod timeutil;

pub use base::base_arena::Allocator;
pub use base::base_comm::{Receiver, Sender};
pub use base::base_memory::{
    DEFAULT_HARD_LIMIT_BYTES, DEFAULT_SAMPLE_INTERVAL, DEFAULT_SOFT_LIMIT_BYTES,
    EditorMemoryStats, EditStatsProvider, MemoryBudget, MemoryCheckSnapshot,
    MemoryMonitor, MemoryPressure, MemorySampleCallback, PeriodicMemoryChecker,
    SystemMemory, sample_system_memory,
};
pub use base::base_task::{Daemon, DaemonError, DaemonState, Runnable};

pub use edit::wredit_base::{WritableAlignedPtr, WritableGapSize};
pub use edit::wredit_cursor::{BidiIndex, Cursor};
pub use edit::wredit_encode::{Encoder, EncodingType};

#[cfg(target_os = "android")]
pub use android::{
    CallbackManager, InputCallback, InputEvent, KeyAction, KeyEvent, MouseEvent,
    MouseEventType,
};
