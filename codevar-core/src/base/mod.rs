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

//! Foundational utilities: arena allocation, channels, tasks, and memory pressure.

pub mod base_arena;
pub mod base_comm;
pub mod base_memory;
pub use base_arena::Allocator;
pub mod base_task;

pub use base_comm::{Receiver, Sender};
pub use base_memory::{
    DEFAULT_HARD_LIMIT_BYTES, DEFAULT_SAMPLE_INTERVAL, DEFAULT_SOFT_LIMIT_BYTES,
    EditorMemoryStats, EditStatsProvider, MemoryBudget, MemoryCheckSnapshot,
    MemoryMonitor, MemoryPressure, MemorySampleCallback, PeriodicMemoryChecker,
    SystemMemory, sample_system_memory,
};
pub use base_task::{Daemon, DaemonError, DaemonState, Runnable};
