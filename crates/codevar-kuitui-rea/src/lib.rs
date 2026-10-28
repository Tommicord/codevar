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
//! 4. **Reconciliation** — children are matched by key and type, first in
//!    order and then through a map keyed by key or position, which keeps the
//!    diff linear in the number of children and preserves identity when a
//!    list is reordered.
//! 5. **Commit** — the finished tree is applied to the host in three phases
//!    (mutation, layout effects, passive effects). Mutation is the only phase
//!    allowed to change the host tree; the root swaps its current tree between
//!    mutation and layout.
//!
//! # Status
//!
//! The crate is being built bottom-up. The lane model, the host boundary, the
//! fiber arena, the hook model, the element vocabulary, keyed reconciliation,
//! the begin/complete work pass, the commit phases and the root that drives
//! them are in place. The public surface is [`lane`], [`element`], [`Host`],
//! [`Key`], [`Props`], [`RenderCtx`], [`State`], [`Cleanup`] and [`Root`].
//!
//! The root renders every pending lane before it returns by default, and a
//! render whose lanes are speculative can be time sliced: [`Root::flush_slice`]
//! stops between units of work and resumes the tree on the next call, while
//! work that is strictly more urgent throws the tree away and starts over. A
//! lane that waits past its expiration delay renders without slicing, so urgent
//! work cannot starve it. A component that writes to its own state while it
//! renders is rendered again in place, so the intermediate tree is never
//! mounted, and a fiber whose description did not change and whose lanes the
//! render does not cover bails out. There is no suspension yet.
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

pub mod element;
pub mod lane;

mod commit;
mod fiber;
mod hooks;
mod host;
mod reconcile;
mod root;
mod wakeable;
mod work;

pub use element::Element;
pub use hooks::{Cleanup, RenderCtx, State};
pub use host::{Host, Key, Props, props, props_ref};
pub use lane::{LANE_COUNT, Lanes, SchedulerPriority};
pub use root::Root;
pub use wakeable::{Wake, Wakeable};
