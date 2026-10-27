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

//! The hook model.
//!
//! A component describes state, memoized values and effects with the methods
//! of [`RenderCtx`]. The hooks are stored *positionally* on the fiber: the
//! first `use_state` of a render owns slot 0, the second owns slot 1, and so
//! on. The same call must therefore sit at the same place on every render —
//! the usual rules of hooks. The engine cannot enforce that with types, so it
//! truncates extra slots and leaves a wrong order to fail at the type downcast
//! of the hook.
//!
//! # Why a render scope
//!
//! React routes `useState` through a module-global dispatcher swapped before
//! every component. Rust needs neither the indirection nor the global: hooks
//! are methods on [`RenderCtx`], the value the host calls the component with.
//! The context carries the fiber, the render's lanes and the update mailbox,
//! so a hook can enqueue an update without knowing where the root lives.
//!
//! # Update queues
//!
//! An update is `(lane, action)`. The action is either a value or a function
//! of the previous value. Functions are stored as `Fn` rather than `FnOnce`
//! so an interrupted render can replay them without losing the update, and
//! updates are only removed from the queue when the tree commits — the
//! committed alternate is the one that advances the queue.
//!
//! A render stops at the first queued update whose lane is not part of the
//! render, leaving it and everything after it for a later render. React
//! rebases such an update and keeps processing; stopping is simpler and never
//! applies an update out of order.
//!
//! # Equality
//!
//! When the queue is empty, [`State::set`] and [`State::update`] compute the
//! next value immediately. If it equals the last committed value, the update
//! is queued with no lane and the root is not woken: the render would produce
//! the same tree. The comparison is only sound while the queue is empty,
//! because only then is the committed value the base of the update.

// The work loop that drives these hooks lands in the next increment.
#![allow(dead_code)]

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::any::Any;
use core::cell::{Cell, RefCell};
use core::mem;

use crate::fiber::FiberId;
use crate::lane::Lanes;

/// The cleanup of an effect.
///
/// The engine runs it before the effect runs again and when the component
/// unmounts. Returning `None` from the effect's create function means there is
/// nothing to clean up.
pub type Cleanup = Box<dyn FnOnce()>;

/// What a dispatch does to the state.
enum UpdateAction<T> {
    /// Replace the value.
    Set(Rc<T>),
    /// Compute the next value from the previous one.
    Apply(Rc<dyn Fn(&T) -> T>),
}

/// One queued update of a state hook.
struct Update {
    /// The lane the update arrived on. [`Lanes::NONE`] marks an eagerly
    /// discarded update: it is kept for rebasing but never wakes the root.
    lane: Lanes,
    /// An `UpdateAction<T>`, type-erased until the render knows `T`.
    action: Box<dyn Any>,
}

/// The queue shared by the alternates of a fiber.
pub(crate) struct UpdateQueue {
    /// The updates not committed yet.
    updates: Vec<Update>,
    /// The fiber of the current tree that owns the queue.
    ///
    /// A dispatch closure outlives the render that created it, so it cannot
    /// capture a fiber id: the id of the current tree changes when alternates
    /// swap. The owner is refreshed by every render and by the commit.
    owner: Cell<FiberId>,
    /// The last committed value, as `Rc<T>`.
    last_rendered: Option<Rc<dyn Any>>,
}

/// The signal a dispatch raises for the root.
#[derive(Default)]
pub(crate) struct Mailbox {
    /// Fibers waiting for a render, with the lanes of their updates.
    updates: Vec<(FiberId, Lanes)>,
}

impl Mailbox {
    /// Records that `fiber` has work on `lanes`.
    pub(crate) fn signal(&mut self, fiber: FiberId, lanes: Lanes) {
        self.updates.push((fiber, lanes));
    }

    /// Takes the work collected since the last drain.
    pub(crate) fn drain(&mut self) -> Vec<(FiberId, Lanes)> {
        mem::take(&mut self.updates)
    }

    /// Whether no dispatch happened since the last drain.
    pub(crate) fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }
}

/// A shared handle to the root's mailbox.
pub(crate) type MailboxHandle = Rc<RefCell<Mailbox>>;

/// Creates an empty mailbox.
pub(crate) fn mailbox() -> MailboxHandle {
    Rc::new(RefCell::new(Mailbox::default()))
}

/// A value that survives a recomposition and can be written from anywhere.
///
/// The state is a handle, not a value: cloning it is cheap and every clone
/// observes the same updates. Updating it queues an update in the hook that
/// created it and wakes the root; the component sees the new value on the next
/// render.
pub struct State<T> {
    /// The value this render reads.
    value: Rc<T>,
    /// The queue of the creating hook.
    queue: Rc<RefCell<UpdateQueue>>,
    /// The root's mailbox.
    mailbox: MailboxHandle,
}

impl<T> State<T> {
    /// The value of the state in this render.
    #[must_use]
    pub fn get(&self) -> &T {
        &self.value
    }
}

impl<T: PartialEq + 'static> State<T> {
    /// Replaces the value on the default lane.
    pub fn set(&self, value: T) {
        self.set_with_lane(Lanes::DEFAULT, value);
    }

    /// Replaces the value on `lane`.
    pub fn set_with_lane(&self, lane: Lanes, value: T) {
        if let Some(current) = self.current_value() {
            let next = Rc::new(value);
            if *current == *next {
                self.enqueue(Lanes::NONE, UpdateAction::Set(next));
                return;
            }
            self.enqueue(lane, UpdateAction::Set(next));
            return;
        }
        self.enqueue(lane, UpdateAction::Set(Rc::new(value)));
    }

    /// Computes the new value from the previous one on the default lane.
    pub fn update(&self, update: impl Fn(&T) -> T + 'static) {
        self.update_with_lane(Lanes::DEFAULT, update);
    }

    /// Computes the new value from the previous one on `lane`.
    pub fn update_with_lane(&self, lane: Lanes, update: impl Fn(&T) -> T + 'static) {
        if let Some(current) = self.current_value() {
            let next = Rc::new(update(&current));
            if *current == *next {
                self.enqueue(Lanes::NONE, UpdateAction::Set(next));
                return;
            }
            self.enqueue(lane, UpdateAction::Set(next));
            return;
        }
        self.enqueue(lane, UpdateAction::Apply(Rc::new(update)));
    }

    /// The committed value, when the queue is empty.
    ///
    /// Outside that case the eager computation is skipped: the committed value
    /// is not the base the update would apply to.
    fn current_value(&self) -> Option<Rc<T>> {
        let queue = self.queue.borrow();
        if !queue.updates.is_empty() {
            return None;
        }
        queue
            .last_rendered
            .as_deref()
            .and_then(|value| value.downcast_ref::<Rc<T>>())
            .cloned()
    }

    /// Queues an action and wakes the root when the lane is not empty.
    fn enqueue(&self, lane: Lanes, action: UpdateAction<T>) {
        let owner = {
            let mut queue = self.queue.borrow_mut();
            queue.updates.push(Update {
                lane,
                action: Box::new(action),
            });
            queue.owner.get()
        };
        if !lane.is_empty() {
            self.mailbox.borrow_mut().signal(owner, lane);
        }
    }
}

impl<T> Clone for State<T> {
    fn clone(&self) -> Self {
        Self {
            value: Rc::clone(&self.value),
            queue: Rc::clone(&self.queue),
            mailbox: Rc::clone(&self.mailbox),
        }
    }
}

/// Whether an effect runs after the host mutation or after the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EffectTag {
    /// Runs right after the host tree is mutated, before the frame is
    /// presented.
    Layout,
    /// Runs after the frame, in the passive phase.
    Passive,
}

/// One effect that must run in the upcoming commit.
///
/// The destroy cell is shared by every record the hook produced, so replacing
/// the effect with new dependencies keeps the cleanup of the previous run.
#[derive(Clone)]
pub(crate) struct EffectRecord {
    /// When the effect runs.
    tag: EffectTag,
    /// The function that installs the effect.
    create: Rc<dyn Fn() -> Option<Cleanup>>,
    /// The cleanup installed by the previous run, if any.
    destroy: Rc<RefCell<Option<Cleanup>>>,
    /// The dependencies the effect was created with.
    deps: Option<Rc<dyn Any>>,
}

impl EffectRecord {
    /// Runs the effect and keeps its cleanup as the current one.
    ///
    /// The cleanup of the previous run is not called here: the commit runs it
    /// in the mutation phase for a layout effect and in the passive unmount
    /// pass for a passive one, before any effect of the same tag is created.
    pub(crate) fn create(&self) {
        let cleanup = (self.create)();
        *self.destroy.borrow_mut() = cleanup;
    }

    /// Runs the cleanup of the previous run, if there is one.
    pub(crate) fn unmount(&self) {
        let destroy = self.destroy.borrow_mut().take();
        if let Some(destroy) = destroy {
            destroy();
        }
    }

    /// When the effect runs.
    pub(crate) const fn tag(&self) -> EffectTag {
        self.tag
    }
}

/// The hooks of one fiber.
///
/// The list is cloned shallowly into the work-in-progress fiber: the values
/// are `Rc`s and the queues are shared, so an alternate costs one `Rc` clone
/// per hook and never copies a state value.
#[derive(Clone, Default)]
pub(crate) struct HookList {
    /// One slot per hook.
    slots: Vec<HookSlot>,
}

impl HookList {
    /// Makes the rendered state the committed state.
    ///
    /// Removes the updates the render applied, remembers the value for the
    /// eager comparison and starts the next render from the new value.
    pub(crate) fn commit(&mut self) {
        for slot in &mut self.slots {
            let Some(queue) = &slot.queue else {
                continue;
            };
            let mut queue = queue.borrow_mut();
            let applied = slot.consumed.min(queue.updates.len());
            queue.updates.drain(..applied);
            slot.consumed = 0;
            queue.last_rendered = slot.value.clone();
        }
    }

    /// Points every queue at the fiber that now owns it.
    pub(crate) fn reset_owners(&self, owner: FiberId) {
        for slot in &self.slots {
            if let Some(queue) = &slot.queue {
                queue.borrow_mut().owner.set(owner);
            }
        }
    }

    /// Runs the cleanup of every effect, for a fiber that is being removed.
    pub(crate) fn destroy_effects(&self) {
        for slot in &self.slots {
            if let Some(effect) = &slot.effect {
                effect.unmount();
            }
        }
    }

    /// Drops the extra slots of a render that used fewer hooks than the last.
    fn finish(&mut self, cursor: usize) {
        self.slots.truncate(cursor);
    }

    /// The slot a render reaches next, creating it on the first render.
    ///
    /// The cursor is passed in rather than read from the list so the render
    /// context can keep it next to the other per-render fields.
    fn slot_at(&mut self, cursor: &mut usize) -> &mut HookSlot {
        let index = *cursor;
        *cursor += 1;
        if index == self.slots.len() {
            self.slots.push(HookSlot::default());
        }
        &mut self.slots[index]
    }
}

/// One hook.
#[derive(Clone, Default)]
struct HookSlot {
    /// The value the hook renders: an `Rc<T>` for state and memo, an
    /// `Rc<RefCell<T>>` for a ref, a `u64` for an id.
    value: Option<Rc<dyn Any>>,
    /// The dependencies of a memo hook.
    deps: Option<Rc<dyn Any>>,
    /// The update queue of a state hook, shared by both alternates.
    queue: Option<Rc<RefCell<UpdateQueue>>>,
    /// How many updates of the queue this alternate applied.
    consumed: usize,
    /// The effect of an effect hook, if it must run.
    effect: Option<EffectRecord>,
}

/// The scope a component renders in.
///
/// The context is passed to the component by the work loop and is only valid
/// for one render: holding it across an `await` or storing it in a struct is a
/// borrow error, which is the Rust analogue of React's "hooks can only be
/// called during rendering".
pub struct RenderCtx<'a> {
    /// The fiber being rendered.
    fiber: FiberId,
    /// The fiber's hook storage.
    hooks: &'a mut HookList,
    /// The next free hook slot.
    cursor: usize,
    /// The effects this render must commit.
    effects: &'a mut Vec<EffectRecord>,
    /// The lanes this render was started for.
    render_lanes: Lanes,
    /// The root's mailbox.
    mailbox: MailboxHandle,
    /// The counter of `use_id`.
    ids: &'a mut u64,
    /// Whether a state hook applied an update.
    did_update: bool,
}

impl<'a> RenderCtx<'a> {
    /// Creates the context for one render.
    pub(crate) fn new(
        fiber: FiberId,
        hooks: &'a mut HookList,
        effects: &'a mut Vec<EffectRecord>,
        render_lanes: Lanes,
        mailbox: MailboxHandle,
        ids: &'a mut u64,
    ) -> Self {
        Self {
            fiber,
            hooks,
            cursor: 0,
            effects,
            render_lanes,
            mailbox,
            ids,
            did_update: false,
        }
    }

    /// Ends the render, dropping the slots a shorter render did not use.
    pub(crate) fn finish(&mut self) {
        self.hooks.finish(self.cursor);
    }

    /// Whether a state hook applied an update during this render.
    pub(crate) const fn did_update(&self) -> bool {
        self.did_update
    }

    /// The next hook slot, creating it on the first render.
    fn next_slot<'b>(hooks: &'b mut HookList, cursor: &mut usize) -> &'b mut HookSlot {
        hooks.slot_at(cursor)
    }

    /// Returns a state that survives a recomposition.
    ///
    /// `init` runs once, when the hook is first reached. Writing to the state
    /// queues an update and wakes the root; the component sees the new value
    /// on its next render.
    pub fn use_state<T: 'static>(&mut self, init: impl FnOnce() -> T) -> State<T> {
        let slot = Self::next_slot(self.hooks, &mut self.cursor);
        if let Some(current) = slot
            .value
            .as_deref()
            .and_then(|value| value.downcast_ref::<Rc<T>>())
            && let Some(queue) = &slot.queue
        {
            let queue = Rc::clone(queue);
            let mut value = Rc::clone(current);
            queue.borrow_mut().owner.set(self.fiber);
            let mut applied = slot.consumed;
            {
                let queue = queue.borrow();
                while applied < queue.updates.len() {
                    let update = &queue.updates[applied];
                    if !update.lane.is_subset_of(self.render_lanes) {
                        break;
                    }
                    if let Some(action) = update.action.downcast_ref::<UpdateAction<T>>() {
                        value = match action {
                            UpdateAction::Set(next) => Rc::clone(next),
                            UpdateAction::Apply(apply) => Rc::new(apply(&value)),
                        };
                    }
                    applied += 1;
                }
            }
            if applied > slot.consumed {
                self.did_update = true;
                slot.consumed = applied;
                slot.value = Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>);
            }
            return State {
                value,
                queue,
                mailbox: Rc::clone(&self.mailbox),
            };
        }

        let value = Rc::new(init());
        let queue = Rc::new(RefCell::new(UpdateQueue {
            updates: Vec::new(),
            owner: Cell::new(self.fiber),
            last_rendered: Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>),
        }));
        slot.value = Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>);
        slot.queue = Some(Rc::clone(&queue));
        slot.consumed = 0;
        State {
            value,
            queue,
            mailbox: Rc::clone(&self.mailbox),
        }
    }

    /// Returns a cell that keeps its value across renders.
    ///
    /// Writing to the cell does not re-render; use it for values the component
    /// reads from callbacks, not for values the tree displays.
    pub fn use_ref<T: 'static>(&mut self, init: impl FnOnce() -> T) -> Rc<RefCell<T>> {
        let slot = Self::next_slot(self.hooks, &mut self.cursor);
        if let Some(cell) = slot
            .value
            .as_deref()
            .and_then(|value| value.downcast_ref::<Rc<RefCell<T>>>())
        {
            return Rc::clone(cell);
        }
        let cell = Rc::new(RefCell::new(init()));
        slot.value = Some(Rc::new(Rc::clone(&cell)) as Rc<dyn Any>);
        cell
    }

    /// Computes a value only when its dependencies change.
    ///
    /// Use the empty tuple `()` for a value that is computed once, and a tuple
    /// of the inputs for a value that follows them. An empty dependency list
    /// is not "no dependencies": `()` compares equal on every render.
    pub fn use_memo<T: 'static, D: PartialEq + 'static>(
        &mut self,
        deps: D,
        init: impl FnOnce() -> T,
    ) -> Rc<T> {
        let slot = Self::next_slot(self.hooks, &mut self.cursor);
        let unchanged = slot
            .deps
            .as_deref()
            .and_then(|stored| stored.downcast_ref::<D>())
            .is_some_and(|stored| stored == &deps);
        if !unchanged {
            let value = Rc::new(init());
            slot.value = Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>);
            slot.deps = Some(Rc::new(deps));
            return value;
        }
        match slot
            .value
            .as_deref()
            .and_then(|value| value.downcast_ref::<Rc<T>>())
        {
            Some(value) => Rc::clone(value),
            None => Rc::new(init()),
        }
    }

    /// Returns a callback only when its dependencies change.
    pub fn use_callback<F: 'static, D: PartialEq + 'static>(&mut self, deps: D, callback: F) -> Rc<F> {
        self.use_memo(deps, || callback)
    }

    /// Runs an effect after the host tree is mutated.
    ///
    /// The effect runs when the hook is first reached and whenever `deps`
    /// change. The cleanup returned by the previous run runs first; returning
    /// `None` means there is nothing to clean up.
    pub fn use_layout_effect<D: PartialEq + 'static>(
        &mut self,
        deps: D,
        create: impl Fn() -> Option<Cleanup> + 'static,
    ) {
        self.use_effect_impl(EffectTag::Layout, deps, create);
    }

    /// Runs an effect after the frame.
    ///
    /// Passive effects are the ones an application uses to talk to the outside
    /// world; they may run a moment after the pixels are shown.
    pub fn use_effect<D: PartialEq + 'static>(
        &mut self,
        deps: D,
        create: impl Fn() -> Option<Cleanup> + 'static,
    ) {
        self.use_effect_impl(EffectTag::Passive, deps, create);
    }

    fn use_effect_impl<D: PartialEq + 'static>(
        &mut self,
        tag: EffectTag,
        deps: D,
        create: impl Fn() -> Option<Cleanup> + 'static,
    ) {
        let slot = Self::next_slot(self.hooks, &mut self.cursor);
        let unchanged = slot
            .effect
            .as_ref()
            .and_then(|effect| effect.deps.as_deref())
            .and_then(|stored| stored.downcast_ref::<D>())
            .is_some_and(|stored| stored == &deps);
        if unchanged {
            return;
        }
        let destroy = match &slot.effect {
            Some(effect) => Rc::clone(&effect.destroy),
            None => Rc::new(RefCell::new(None)),
        };
        let record = EffectRecord {
            tag,
            create: Rc::new(create),
            destroy,
            deps: Some(Rc::new(deps)),
        };
        slot.effect = Some(record.clone());
        self.effects.push(record);
    }

    /// Returns a stable identifier for this hook.
    ///
    /// The id is unique within the composition and does not change across
    /// renders, which is what an accessibility attribute or a text-input
    /// association needs.
    pub fn use_id(&mut self) -> u64 {
        let slot = Self::next_slot(self.hooks, &mut self.cursor);
        if let Some(id) = slot
            .value
            .as_deref()
            .and_then(|value| value.downcast_ref::<u64>())
        {
            return *id;
        }
        let id = *self.ids;
        *self.ids = self.ids.wrapping_add(1);
        slot.value = Some(Rc::new(id) as Rc<dyn Any>);
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    struct Harness {
        fiber: FiberId,
        hooks: HookList,
        effects: Vec<EffectRecord>,
        mailbox: MailboxHandle,
        ids: u64,
        lanes: Lanes,
    }

    impl Harness {
        fn new() -> Self {
            Self {
                fiber: FiberId::from_index(0),
                hooks: HookList::default(),
                effects: Vec::new(),
                mailbox: mailbox(),
                ids: 0,
                lanes: Lanes::DEFAULT,
            }
        }

        fn render<R>(&mut self, content: impl FnOnce(&mut RenderCtx<'_>) -> R) -> R {
            // A render consumes the queued marks, the way the root does when it
            // turns a dispatch into pending lanes.
            self.mailbox.borrow_mut().drain();
            self.effects.clear();
            let mut ctx = RenderCtx::new(
                self.fiber,
                &mut self.hooks,
                &mut self.effects,
                self.lanes,
                Rc::clone(&self.mailbox),
                &mut self.ids,
            );
            let result = content(&mut ctx);
            ctx.finish();
            result
        }

        fn signals(&self) -> usize {
            self.mailbox.borrow().updates.len()
        }

        fn commit(&mut self) {
            self.hooks.commit();
        }
    }

    #[test]
    fn a_state_survives_a_render() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        assert_eq!(*state.get(), 1);
        let state = harness.render(|ctx| ctx.use_state(|| 99_u32));
        assert_eq!(*state.get(), 1);
    }

    #[test]
    fn a_dispatch_is_applied_on_the_next_render() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        state.set(2);
        assert_eq!(harness.signals(), 1);
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        assert_eq!(*state.get(), 2);
        harness.commit();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        assert_eq!(*state.get(), 2);
    }

    #[test]
    fn an_equal_update_does_not_wake_the_root() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        harness.commit();
        state.set(1);
        assert_eq!(harness.signals(), 0);
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        assert_eq!(*state.get(), 1);
    }

    #[test]
    fn an_update_waits_for_its_lane() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        state.set_with_lane(Lanes::TRANSITION, 2);
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        assert_eq!(*state.get(), 1);
        harness.lanes = Lanes::TRANSITION;
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        assert_eq!(*state.get(), 2);
    }

    #[test]
    fn a_functional_update_is_replayable() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        state.update(|value| value + 1);
        state.update(|value| value * 3);

        let mut first = harness.hooks.clone();
        let mut second = harness.hooks.clone();
        let mut effects = Vec::new();
        let mut ids = harness.ids;
        let read = |hooks: &mut HookList, effects: &mut Vec<EffectRecord>, ids: &mut u64| {
            let mut ctx = RenderCtx::new(
                harness.fiber,
                hooks,
                effects,
                Lanes::DEFAULT,
                Rc::clone(&harness.mailbox),
                ids,
            );
            let state = ctx.use_state(|| 0_u32);
            ctx.finish();
            *state.get()
        };
        let once = read(&mut first, &mut effects, &mut ids);
        let twice = read(&mut second, &mut effects, &mut ids);
        assert_eq!(once, 6);
        assert_eq!(twice, 6);
    }

    #[test]
    fn a_ref_keeps_its_cell() {
        let mut harness = Harness::new();
        let cell = harness.render(|ctx| ctx.use_ref(|| 3_u32));
        *cell.borrow_mut() = 4;
        let again = harness.render(|ctx| ctx.use_ref(|| 0_u32));
        assert_eq!(*again.borrow(), 4);
    }

    #[test]
    fn a_memo_recomputes_only_when_deps_change() {
        let mut harness = Harness::new();
        let calls = Rc::new(Cell::new(0));
        let counted = Rc::clone(&calls);
        let value = harness.render(|ctx| {
            ctx.use_memo((1_u32,), || {
                counted.set(counted.get() + 1);
                10_u32
            })
        });
        assert_eq!(*value, 10);
        assert_eq!(calls.get(), 1);

        let counted = Rc::clone(&calls);
        let value = harness.render(|ctx| {
            ctx.use_memo((1_u32,), || {
                counted.set(counted.get() + 1);
                20_u32
            })
        });
        assert_eq!(*value, 10);
        assert_eq!(calls.get(), 1);

        let counted = Rc::clone(&calls);
        let value = harness.render(|ctx| {
            ctx.use_memo((2_u32,), || {
                counted.set(counted.get() + 1);
                30_u32
            })
        });
        assert_eq!(*value, 30);
        assert_eq!(calls.get(), 2);
    }

    fn seven() -> u32 {
        7
    }

    fn eight() -> u32 {
        8
    }

    #[test]
    fn a_callback_is_stable_across_renders() {
        let mut harness = Harness::new();
        let first = harness.render(|ctx| ctx.use_callback((), seven as fn() -> u32));
        let second = harness.render(|ctx| ctx.use_callback((), eight as fn() -> u32));
        assert!(Rc::ptr_eq(&first, &second));
        assert_eq!(second(), 7);
    }

    #[test]
    fn an_effect_runs_when_its_deps_change() {
        let mut harness = Harness::new();
        let runs = Rc::new(Cell::new(0));
        let counted = Rc::clone(&runs);
        harness.render(move |ctx| {
            ctx.use_effect((), move || {
                counted.set(counted.get() + 1);
                None
            })
        });
        assert_eq!(harness.effects.len(), 1);
        assert_eq!(harness.effects[0].tag(), EffectTag::Passive);

        harness.render(|ctx| ctx.use_effect((), || None));
        assert!(harness.effects.is_empty());

        let counted = Rc::clone(&runs);
        harness.render(move |ctx| {
            ctx.use_effect((1_u32,), move || {
                counted.set(counted.get() + 1);
                None
            })
        });
        assert_eq!(harness.effects.len(), 1);
        assert_eq!(harness.effects[0].tag(), EffectTag::Passive);
    }

    #[test]
    fn a_layout_effect_keeps_the_previous_cleanup() {
        let mut harness = Harness::new();
        let cleaned = Rc::new(Cell::new(0));
        let first = Rc::clone(&cleaned);
        harness.render(move |ctx| {
            ctx.use_layout_effect((), move || {
                let cleaned = Rc::clone(&first);
                Some(Box::new(move || cleaned.set(cleaned.get() + 1)))
            })
        });
        assert_eq!(harness.effects.len(), 1);
        harness.effects[0].unmount();
        harness.effects[0].create();
        assert_eq!(cleaned.get(), 0);

        let second = Rc::clone(&cleaned);
        harness.render(move |ctx| {
            ctx.use_layout_effect((1_u32,), move || {
                let cleaned = Rc::clone(&second);
                Some(Box::new(move || cleaned.set(cleaned.get() + 1)))
            })
        });
        assert_eq!(harness.effects.len(), 1);
        harness.effects[0].unmount();
        harness.effects[0].create();
        assert_eq!(cleaned.get(), 1);
    }

    #[test]
    fn an_id_is_stable_and_unique() {
        let mut harness = Harness::new();
        let (first, second) = harness.render(|ctx| (ctx.use_id(), ctx.use_id()));
        assert_ne!(first, second);
        let again = harness.render(|ctx| (ctx.use_id(), ctx.use_id()));
        assert_eq!((first, second), again);
    }

    #[test]
    fn commit_drops_applied_updates_and_advances_the_eager_base() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 0_u32));
        state.set(1);
        let state = harness.render(|ctx| ctx.use_state(|| 0_u32));
        assert_eq!(*state.get(), 1);
        harness.commit();
        state.set(1);
        assert_eq!(harness.signals(), 0);
    }

    #[test]
    fn a_shorter_render_drops_the_extra_slots() {
        let mut harness = Harness::new();
        harness.render(|ctx| {
            ctx.use_state(|| 1_u32);
            ctx.use_state(|| 2_u32);
        });
        harness.render(|ctx| {
            ctx.use_state(|| 3_u32);
        });
        assert_eq!(harness.hooks.slots.len(), 1);
    }
}
