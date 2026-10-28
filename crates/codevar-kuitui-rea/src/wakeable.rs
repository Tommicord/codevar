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

//! The value a component hands back when it suspends.
//!
//! React's component throws a promise and the engine subscribes to it with
//! `then` to learn that the work completed. Rust has no exceptions, so a
//! component calls [`RenderCtx::suspend`](crate::RenderCtx::suspend) and
//! returns its value instead; the work it is waiting on travels as a
//! [`Wakeable`]. Everything else is the same: the engine carries the wakeable
//! to the nearest Suspense boundary, and the boundary retries its primary tree
//! once the work reports that it completed.
//!
//! A wakeable is a handle, not a future. It does not own the work and it does
//! not know what the work produces — it is only the callback registration the
//! engine needs, which is all React asks of a thenable. The same type covers
//! a network request, a thread, or a test that resolves by hand:
//!
//! ```
//! use codevar_kuitui_rea::{Wakeable, Wake};
//! use std::rc::Rc;
//! use std::cell::RefCell;
//!
//! let woken = Rc::new(RefCell::new(false));
//! let flag = Rc::clone(&woken);
//! let wakeable = Wakeable::from_fn(move |wake| {
//!     *flag.borrow_mut() = true;
//!     wake();
//! });
//! wakeable.subscribe(Rc::new(|| {}));
//! assert!(*woken.borrow());
//! ```

use alloc::rc::Rc;
use core::fmt;

/// What the engine subscribes to when a component suspends.
///
/// Implement it for the handle of the work that suspended, or use the blanket
/// implementation for a closure of the shape `Fn(Rc<dyn Fn()>)` and build the
/// wakeable with [`Wakeable::from_fn`].
///
/// # Implementor contract
///
/// * `subscribe` must run `wake` at most once, when the work completes. Running
///   it more often makes the boundary retry more often; it must never run it
///   before returning.
/// * The callback may be registered on a work item that already completed, in
///   which case the implementation must run it immediately.
pub trait Wake {
    /// Registers `wake` to run when the work completes.
    fn subscribe(&self, wake: Rc<dyn Fn()>);
}

impl<F: Fn(Rc<dyn Fn()>) + 'static> Wake for F {
    fn subscribe(&self, wake: Rc<dyn Fn()>) {
        self(wake);
    }
}

/// A handle to the work that suspended a component.
///
/// Cloning the wakeable is cheap and every clone names the same work; two
/// wakeables compare the same only when they wrap the same handle
/// ([`Wakeable::same`]). A wakeable is not `PartialEq`: the identity is a
/// pointer comparison, not a value comparison.
#[derive(Clone)]
pub struct Wakeable(Rc<dyn Wake>);

impl Wakeable {
    /// Wraps a handle the component implements [`Wake`] on.
    pub fn new<W: Wake + 'static>(wake: W) -> Self {
        Self(Rc::new(wake))
    }

    /// Wraps a closure that runs its argument when the work completes.
    pub fn from_fn<F: Fn(Rc<dyn Fn()>) + 'static>(wake: F) -> Self {
        Self(Rc::new(wake))
    }

    /// Registers `wake` for when the work completes.
    ///
    /// The engine uses this when a boundary captures the suspension; an
    /// application may subscribe too, because every subscriber is independent.
    pub fn subscribe(&self, wake: Rc<dyn Fn()>) {
        self.0.subscribe(wake);
    }

    /// Whether both handles name the same work.
    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl fmt::Debug for Wakeable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Wakeable")
            .field("handle", &Rc::as_ptr(&self.0))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::cell::{Cell, RefCell};

    #[test]
    fn a_fn_wakeable_runs_its_subscriber_when_subscribed() {
        let called = Rc::new(Cell::new(0_u32));
        let seen = Rc::clone(&called);
        let wakeable = Wakeable::from_fn(move |wake| {
            seen.set(seen.get() + 1);
            wake();
        });
        let woken = Rc::new(Cell::new(false));
        let flag = Rc::clone(&woken);
        wakeable.subscribe(Rc::new(move || flag.set(true)));
        assert_eq!(called.get(), 1);
        assert!(woken.get());
    }

    #[test]
    fn a_custom_wake_reaches_the_callback() {
        struct Manual {
            subscribers: RefCell<Vec<Rc<dyn Fn()>>>,
        }

        impl Wake for Rc<Manual> {
            fn subscribe(&self, wake: Rc<dyn Fn()>) {
                self.subscribers.borrow_mut().push(wake);
            }
        }

        let manual = Rc::new(Manual {
            subscribers: RefCell::new(Vec::new()),
        });
        let wakeable = Wakeable::new(Rc::clone(&manual));
        let woken = Rc::new(Cell::new(false));
        let flag = Rc::clone(&woken);
        wakeable.subscribe(Rc::new(move || flag.set(true)));
        assert!(!woken.get());
        let subscribers = manual.subscribers.borrow();
        assert_eq!(subscribers.len(), 1);
        if let Some(wake) = subscribers.first() {
            wake();
        }
        assert!(woken.get());
    }

    #[test]
    fn clones_name_the_same_work_and_another_handle_does_not() {
        let first = Wakeable::from_fn(|_| {});
        let clone = first.clone();
        let other = Wakeable::from_fn(|_| {});
        assert!(first.same(&clone));
        assert!(!first.same(&other));
    }
}
