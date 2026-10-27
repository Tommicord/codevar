//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of
//! the License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! Rea — the reactivity and composition engine of Kuitui.
//!
//! Rea is a port of the parts of React that make a declarative UI cheap to
//! update: the *fiber* tree, the hook model, the lane-based scheduler and the
//! three-phase commit. It is deliberately independent from the widget
//! framework: it knows how to reconcile a tree and when to run an effect, but
//! it does not know what a widget is. The host provides that knowledge through
//! a small [`Host`] trait, and `codevar-kuitui` implements it on top of its
//! retained widget tree.
//!
//! The split keeps the engine testable without a GPU, lets a second host — a
//! text-only test harness, a future accessibility tree — reuse the same
//! reconciler, and keeps the widget crate free of scheduling code.
//!
//! # Layers
//!
//! 1. **Lanes** — every unit of work carries a priority lane. The bit position
//!    of a lane is its priority, so the highest-priority pending work is the
//!    lowest set bit. Lanes make interruption, batching, starvation protection
//!    and "render the urgent part first" a matter of bit arithmetic instead of
//!    a second scheduling rule.
//! 2. **Fibers** — the tree is stored in an arena of double-buffered nodes.
//!    A update builds a work-in-progress tree next to the committed one, so an
//!    interrupted render can be thrown away without corrupting what is on
//!    screen, and reused nodes keep their hooks and host state.
//! 3. **Hooks** — components describe state, memoized values and effects with
//!    positional hooks (`use_state`, `use_memo`, `use_effect`, ...). The hook
//!    list of a fiber survives a render, and its update queues are shared with
//!    the work-in-progress clone, exactly as React shares them between a fiber
//!    and its alternate.
//! 4. **Reconciliation** — children are matched by key and type in one pass
//!    over a hash map, which keeps the diff linear in the number of children
//!    and preserves identity when a list is reordered.
//! 5. **Commit** — the finished tree is applied to the host in three phases
//!    (mutation, layout effects, passive effects). Mutation is the only phase
//!    allowed to change the host tree; the root swaps its current tree between
//!    mutation and layout.
//!
//! # Status
//!
//! The crate is being built bottom-up. The lane model, the host boundary and
//! the fiber arena are in place; hooks, reconciliation, the work loop and the
//! commit phase follow. Until the root lands, the public surface is
//! [`lane`], [`Host`], [`Key`] and [`Props`].
//!
//! # References
//!
//! The invariants implemented here follow the React reconciler
//! (`ReactFiberHooks`, `ReactFiberWorkLoop`, `ReactFiberLane`,
//! `ReactChildFiber`, `ReactFiberCommitWork`). Where Rust forces a different
//! shape — hooks are methods on a render scope rather than free functions, and
//! the host is a generic trait rather than a global module — the difference is
//! documented at the point where it matters.

#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod lane;

mod fiber;
mod host;

pub use host::{Host, Key, Props, props, props_ref};
pub use lane::{LANE_COUNT, Lanes, SchedulerPriority};
