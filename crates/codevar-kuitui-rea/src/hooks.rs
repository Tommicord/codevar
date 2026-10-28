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
//! so a skipped update can be replayed by a later render without losing it,
//! and an applied update leaves the queue when a render applies it.
//!
//! A render applies every update whose lane is part of the render's lanes and
//! *rebases* the rest, the way React's `updateReducer` does. The updates a
//! render skipped stay in a base queue with their lanes, together with a copy
//! of every update it applied after the skip; the copy has no lane, so a later
//! render replays the chunk in source order no matter which lanes it covers.
//! The skipped lane is also recorded on the fiber, which is what makes the
//! root render that lane again.
//!
//! # Equality
//!
//! When the queue is empty, [`State::set`] and [`State::update`] compute the
//! next value immediately. If it equals the last rendered value, the update
//! is queued with no lane and the root is not woken: the render would produce
//! the same tree. The comparison is only sound while the queue is empty,
//! because only then is the last rendered value the base of the update.
//!
//! # Render phase updates
//!
//! A component that writes to its own state while it renders is a *render
//! phase update*. React does not schedule a new render for it: the update
//! applies to the component that is rendering, and that component renders
//! again in place before the pass goes on, so the tree the commit sees is the
//! one the final value produced and the intermediate tree is never mounted.
//!
//! The mailbox records which component is rendering, and every state handle
//! already shares it. A dispatch on a queue that component owns is tagged with
//! the lanes of the render instead of its own lanes and does not wake the
//! root; the work pass reruns the component and the queue replays the update
//! like any other. A dispatch on any other queue — a parent writing a child's
//! state, say — is an ordinary update and takes the usual path.

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
    /// The lane the update arrived on. [`Lanes::NONE`] marks an update that
    /// must be replayed by the next render whatever its lanes are: it was
    /// either queued by the eager comparison, which already knows it changes
    /// nothing, or rebased on top of a skipped update.
    lane: Lanes,
    /// An `UpdateAction<T>`, type-erased until the render knows `T`.
    ///
    /// The action is shared rather than moved so a rebase can copy the update
    /// without copying the closure.
    action: Rc<dyn Any>,
}

impl Update {
    /// The copy a render keeps when it applied the update on top of a skipped
    /// one.
    ///
    /// The copy keeps the action but drops the lane: React replays it in the
    /// next render regardless of which lanes that render covers, because the
    /// skipped update it was applied to has to be replayed first.
    fn rebased(&self) -> Self {
        Self {
            lane: Lanes::NONE,
            action: Rc::clone(&self.action),
        }
    }
}

/// The queue shared by the alternates of a fiber.
pub(crate) struct UpdateQueue {
    /// The updates queued since the last render.
    pending: Vec<Update>,
    /// The updates a render skipped, in source order, followed by the rebased
    /// copies of the updates it applied after the first skip.
    base: Vec<Update>,
    /// The value the base queue applies to.
    base_state: Option<Rc<dyn Any>>,
    /// The fiber of the current tree that owns the queue, or `None` when the
    /// fiber that created it is gone.
    ///
    /// A dispatch closure outlives the render that created it, so it cannot
    /// capture a fiber id: the id of the current tree changes when alternates
    /// swap. The owner is refreshed by every render and by the commit, and the
    /// commit detaches it when the fiber unmounts. A detached queue keeps the
    /// updates it already holds but no longer wakes the root, which is what
    /// React does when it finds no path from the fiber back to a root.
    owner: Cell<Option<FiberId>>,
    /// The last rendered value, as `Rc<T>`.
    last_rendered: Option<Rc<dyn Any>>,
}

impl UpdateQueue {
    /// The fiber that currently owns the queue, if it is still attached.
    pub(crate) fn owner(&self) -> Option<FiberId> {
        self.owner.get()
    }

    /// Whether no render has work left in the queue.
    fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.base.is_empty()
    }

    /// Removes the pending updates that were queued on `lanes` themselves.
    ///
    /// Only a render phase update carries exactly the lanes of the render it
    /// was queued in, so only those are removed. An update that arrived from
    /// outside keeps its lane and waits for the render that covers it, and an
    /// update with no lane keeps the eager comparison's meaning.
    fn discard(&mut self, lanes: Lanes) {
        self.pending
            .retain(|update| update.lane.is_empty() || !update.lane.is_subset_of(lanes));
    }
}

/// A shared handle to one update queue.
pub(crate) type QueueHandle = Rc<RefCell<UpdateQueue>>;

/// The signal a dispatch raises for the root.
#[derive(Default)]
pub(crate) struct Mailbox {
    /// The queues waiting for a render, with the lanes of their updates.
    ///
    /// The queue, not the fiber, is what a dispatch can name: the fiber id of
    /// the current tree changes when alternates swap, while the queue is
    /// shared by both alternates and knows which fiber owns it right now.
    updates: Vec<(QueueHandle, Lanes)>,
    /// The component render that is in flight, when there is one.
    render_phase: Option<RenderPhase>,
}

/// The component render the root is running right now.
///
/// The record is what turns a dispatch into a render phase update: a queue
/// the rendering fiber owns names the component being rendered, so the update
/// applies to that render instead of scheduling a new one.
struct RenderPhase {
    /// The fiber being rendered.
    fiber: FiberId,
    /// The other half of that fiber.
    ///
    /// A state handle captured before the render began still reaches its
    /// queue through the committed fiber, which is this alternate until the
    /// first hook of the pass refreshes the owner.
    alternate: Option<FiberId>,
    /// The lanes of the render, which tag its updates.
    lanes: Lanes,
    /// Whether a dispatch queued an update since the pass began.
    did_schedule_update: bool,
}

impl RenderPhase {
    /// Whether a queue owned by `owner` belongs to this render.
    fn owns(&self, owner: Option<FiberId>) -> bool {
        owner == Some(self.fiber) || owner == self.alternate
    }
}

impl Mailbox {
    /// Records that the owner of `queue` has work on `lanes`.
    pub(crate) fn signal(&mut self, queue: QueueHandle, lanes: Lanes) {
        self.updates.push((queue, lanes));
    }

    /// Takes the work collected since the last drain.
    pub(crate) fn drain(&mut self) -> Vec<(QueueHandle, Lanes)> {
        mem::take(&mut self.updates)
    }

    /// Whether no dispatch happened since the last drain.
    pub(crate) fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }

    /// Starts the render of a component.
    ///
    /// From here until [`Mailbox::end_render`], a dispatch on a queue the
    /// fiber owns is a render phase update.
    pub(crate) fn begin_render(&mut self, fiber: FiberId, alternate: Option<FiberId>, lanes: Lanes) {
        debug_assert!(
            self.render_phase.is_none(),
            "a component render cannot start inside another"
        );
        self.render_phase = Some(RenderPhase {
            fiber,
            alternate,
            lanes,
            did_schedule_update: false,
        });
    }

    /// Whether a dispatch queued a render phase update since the last call,
    /// clearing the mark.
    ///
    /// Every pass of a component asks for its own mark: it is the signal that
    /// the function must run again before the element is reconciled.
    pub(crate) fn take_render_phase_update(&mut self) -> bool {
        self.render_phase
            .as_mut()
            .is_some_and(|phase| mem::replace(&mut phase.did_schedule_update, false))
    }

    /// Ends the component render.
    pub(crate) fn end_render(&mut self) {
        self.render_phase = None;
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
    queue: QueueHandle,
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
    ///
    /// `lane` is the identity of the update, so it is a single lane, the way a
    /// React update carries one `Lane` and not a mask: a render covers the
    /// update only when the lane is a subset of the lanes it renders.
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
    ///
    /// `lane` is a single lane, as in [`State::set_with_lane`].
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

    /// The last rendered value, when the queue is empty.
    ///
    /// Outside that case the eager computation is skipped: the last rendered
    /// value is not the base the update would apply to.
    fn current_value(&self) -> Option<Rc<T>> {
        let queue = self.queue.borrow();
        if !queue.is_empty() {
            return None;
        }
        queue
            .last_rendered
            .as_deref()
            .and_then(|value| value.downcast_ref::<Rc<T>>())
            .cloned()
    }

    /// Queues an action and wakes the root when the lane is not empty.
    ///
    /// A dispatch on the queue of the component that is rendering right now is
    /// a render phase update: it takes the lanes of the render, so the next
    /// pass applies it, and it does not wake the root, because the render that
    /// is already running replays it before the commit.
    fn enqueue(&self, lane: Lanes, action: UpdateAction<T>) {
        let mut mailbox = self.mailbox.borrow_mut();
        let mut queue = self.queue.borrow_mut();
        let render_phase = if lane.is_empty() {
            // The eager comparison already knows the value cannot change:
            // there is nothing to replay, in a render phase or anywhere else.
            None
        } else {
            mailbox
                .render_phase
                .as_mut()
                .filter(|phase| phase.owns(queue.owner.get()))
        };
        let (lane, signal) = match render_phase {
            Some(phase) => {
                phase.did_schedule_update = true;
                (phase.lanes, false)
            }
            None => (lane, !lane.is_empty() && queue.owner.get().is_some()),
        };
        queue.pending.push(Update {
            lane,
            action: Rc::new(action),
        });
        drop(queue);
        if signal {
            mailbox.signal(Rc::clone(&self.queue), lane);
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
    /// The effect dependencies the committed render recorded, one entry per
    /// hook slot.
    ///
    /// A re-render pass compares its effect dependencies against this
    /// snapshot, not against the slot the previous pass wrote: React rebuilds
    /// the effect list of every pass from the committed hook, so a dependency
    /// that changes between passes is committed once, with the value of the
    /// last pass.
    committed_effects: Vec<Option<Rc<dyn Any>>>,
}

impl HookList {
    /// Freezes the committed effect dependencies for the render that starts
    /// now.
    pub(crate) fn begin_render(&mut self) {
        self.committed_effects = self
            .slots
            .iter()
            .map(|slot| {
                slot.effect
                    .as_ref()
                    .and_then(|effect| effect.deps.as_ref().map(Rc::clone))
            })
            .collect();
    }

    /// The dependencies of a slot's effect in the committed render, when the
    /// committed render reached that slot and the effect had dependencies.
    fn committed_deps(&self, index: usize) -> Option<Rc<dyn Any>> {
        self.committed_effects
            .get(index)
            .cloned()
            .flatten()
    }

    /// Makes the rendered state the committed state.
    ///
    /// The applied updates already left the queue during the render, so the
    /// only thing left to commit is the value the eager comparison starts
    /// from.
    pub(crate) fn commit(&mut self) {
        for slot in &mut self.slots {
            if let Some(queue) = &slot.queue {
                queue.borrow_mut().last_rendered = slot.value.clone();
            }
        }
    }

    /// Points every queue at the fiber that now owns it.
    pub(crate) fn reset_owners(&self, owner: FiberId) {
        for slot in &self.slots {
            if let Some(queue) = &slot.queue {
                queue.borrow_mut().owner.set(Some(owner));
            }
        }
    }

    /// Drops the render phase updates of a component that rendered too often.
    ///
    /// A render phase update applies in the pass that queued it. When a
    /// component asks for more passes than the limit allows, the update of the
    /// last pass has no pass left to apply it, and keeping it in the queue
    /// would apply it out of order in a later render.
    pub(crate) fn discard_render_phase_updates(&self, lanes: Lanes) {
        for slot in &self.slots {
            if let Some(queue) = &slot.queue {
                queue.borrow_mut().discard(lanes);
            }
        }
    }

    /// Detaches every queue of the fiber.
    ///
    /// The commit calls this when the fiber is removed. The state handles the
    /// component handed out may still be alive, but an update through them
    /// would name a fiber that no longer exists, so they stop waking the root.
    pub(crate) fn detach(&self) {
        for slot in &self.slots {
            if let Some(queue) = &slot.queue {
                queue.borrow_mut().owner.set(None);
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
    /// The lanes this render could not apply, for the fiber to record.
    ///
    /// A state hook that skipped an update adds its lane here; the work loop
    /// copies the set onto the fiber so the root knows a later render has to
    /// cover it.
    lanes: &'a mut Lanes,
    /// The root's mailbox.
    mailbox: MailboxHandle,
    /// The counter of `use_id`.
    ids: &'a mut u64,
    /// Whether a state hook applied an update.
    ///
    /// React marks the fiber only when the resulting value differs from the
    /// one this alternate rendered. Comparing the values would need a
    /// `PartialEq` bound on every state type, so the engine records that an
    /// update ran; the difference only matters to the bailout that arrives
    /// with `bailoutOnAlreadyFinishedWork`.
    did_update: bool,
}

impl<'a> RenderCtx<'a> {
    /// Creates the context for one render.
    pub(crate) fn new(
        fiber: FiberId,
        hooks: &'a mut HookList,
        effects: &'a mut Vec<EffectRecord>,
        render_lanes: Lanes,
        lanes: &'a mut Lanes,
        mailbox: MailboxHandle,
        ids: &'a mut u64,
    ) -> Self {
        Self {
            fiber,
            hooks,
            cursor: 0,
            effects,
            render_lanes,
            lanes,
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
            let value = {
                let mut queue = queue.borrow_mut();
                queue.owner.set(Some(self.fiber));
                let (value, applied) =
                    Self::replay::<T>(&mut queue, Rc::clone(current), self.render_lanes, self.lanes);
                self.did_update |= applied;
                value
            };
            slot.value = Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>);
            return State {
                value,
                queue,
                mailbox: Rc::clone(&self.mailbox),
            };
        }

        let value = Rc::new(init());
        let queue = Rc::new(RefCell::new(UpdateQueue {
            pending: Vec::new(),
            base: Vec::new(),
            base_state: Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>),
            owner: Cell::new(Some(self.fiber)),
            last_rendered: Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>),
        }));
        slot.value = Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>);
        slot.queue = Some(Rc::clone(&queue));
        State {
            value,
            queue,
            mailbox: Rc::clone(&self.mailbox),
        }
    }

    /// Replays the update queue of a state hook for one render.
    ///
    /// Returns the value this render reads and whether it applied an update.
    /// The base queue the previous render left is replayed first, then the
    /// updates queued since. An update outside the render's lanes goes back
    /// into the base queue with its lane, and its lane is added to `skipped`;
    /// every update applied after such a skip is copied into the base queue
    /// without a lane, so the next render replays the tail in source order on
    /// top of the skipped update. The committed alternate's value is the base
    /// when the queue has none, which only happens for a queue created before
    /// the base state existed.
    fn replay<T: 'static>(
        queue: &mut UpdateQueue,
        current: Rc<T>,
        render_lanes: Lanes,
        skipped: &mut Lanes,
    ) -> (Rc<T>, bool) {
        let mut updates = mem::take(&mut queue.base);
        updates.append(&mut queue.pending);
        let had_updates = !updates.is_empty();
        let mut value = queue
            .base_state
            .as_deref()
            .and_then(|state| state.downcast_ref::<Rc<T>>())
            .map_or(current, Rc::clone);
        let mut base = Vec::new();
        let mut base_state = None;
        let mut applied = false;
        for update in updates {
            if !update.lane.is_subset_of(render_lanes) {
                if base_state.is_none() {
                    base_state = Some(Rc::clone(&value));
                }
                skipped.insert(update.lane);
                base.push(update);
                continue;
            }
            if base_state.is_some() {
                base.push(update.rebased());
            }
            if let Some(action) = update.action.downcast_ref::<UpdateAction<T>>() {
                value = match action {
                    UpdateAction::Set(next) => Rc::clone(next),
                    UpdateAction::Apply(apply) => Rc::new(apply(&value)),
                };
                applied = true;
            }
        }
        let base_state = base_state.unwrap_or_else(|| Rc::clone(&value));
        queue.base = base;
        queue.base_state = Some(Rc::new(base_state) as Rc<dyn Any>);
        if had_updates {
            queue.last_rendered = Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>);
        }
        (value, applied)
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
        let index = self.cursor;
        let committed = self.hooks.committed_deps(index);
        let unchanged = committed
            .as_deref()
            .and_then(|stored| stored.downcast_ref::<D>())
            .is_some_and(|stored| stored == &deps);
        let slot = Self::next_slot(self.hooks, &mut self.cursor);
        if unchanged {
            // The effect does not run, but the record the work-in-progress
            // hook keeps must describe this pass: a later render compares its
            // dependencies against it, and a re-render may have changed them
            // before this pass put them back.
            if let Some(effect) = &mut slot.effect {
                effect.deps = committed;
            }
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
        skipped: Lanes,
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
                skipped: Lanes::NONE,
            }
        }

        fn render<R>(&mut self, content: impl FnOnce(&mut RenderCtx<'_>) -> R) -> R {
            // A render consumes the queued marks, the way the root does when it
            // turns a dispatch into pending lanes.
            self.mailbox.borrow_mut().drain();
            self.effects.clear();
            self.skipped = Lanes::NONE;
            // The harness has no alternates, so the hooks the render starts
            // from are both the committed and the work-in-progress list.
            self.hooks.begin_render();
            let mut ctx = RenderCtx::new(
                self.fiber,
                &mut self.hooks,
                &mut self.effects,
                self.lanes,
                &mut self.skipped,
                Rc::clone(&self.mailbox),
                &mut self.ids,
            );
            let result = content(&mut ctx);
            ctx.finish();
            result
        }

        /// Starts a component render, so a dispatch from here on is a render
        /// phase update.
        fn begin(&mut self) {
            self.mailbox
                .borrow_mut()
                .begin_render(self.fiber, None, self.lanes);
        }

        fn end(&mut self) {
            self.mailbox.borrow_mut().end_render();
        }

        fn took_update(&mut self) -> bool {
            self.mailbox
                .borrow_mut()
                .take_render_phase_update()
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
    fn an_update_applied_after_a_skip_is_rebased() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        harness.commit();
        state.update_with_lane(Lanes::TRANSITION, |value| value + 10);
        state.update_with_lane(Lanes::DEFAULT, |value| value * 2);

        // The default render skips the transition update, applies the one
        // queued after it, and records the lane it could not cover.
        let state = harness.render(|ctx| ctx.use_state(|| 0_u32));
        assert_eq!(*state.get(), 2);
        assert!(harness.skipped.contains(Lanes::TRANSITION));

        // Another default render replays the same chunk from the same base:
        // the rebased copy is idempotent instead of being applied twice.
        let state = harness.render(|ctx| ctx.use_state(|| 0_u32));
        assert_eq!(*state.get(), 2);

        // The transition render replays the skipped update and then the
        // default update, in the order they were dispatched.
        harness.lanes = Lanes::TRANSITION;
        let state = harness.render(|ctx| ctx.use_state(|| 0_u32));
        assert_eq!(*state.get(), 22);
        assert!(harness.skipped.is_empty());
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
            let mut skipped = Lanes::NONE;
            hooks.begin_render();
            let mut ctx = RenderCtx::new(
                harness.fiber,
                hooks,
                effects,
                Lanes::DEFAULT,
                &mut skipped,
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
    fn an_applied_update_leaves_the_queue_and_advances_the_eager_base() {
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
    fn a_dispatch_names_the_queue_that_owns_the_fiber() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        state.set(2);
        let signals = harness.mailbox.borrow_mut().drain();
        assert_eq!(signals.len(), 1);
        let Some((queue, lanes)) = signals.first() else {
            return;
        };
        assert_eq!(queue.borrow().owner(), Some(harness.fiber));
        assert!(lanes.contains(Lanes::DEFAULT));
    }

    #[test]
    fn a_detached_state_does_not_wake_the_root() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        harness.commit();
        harness.hooks.detach();
        state.set(2);
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

    #[test]
    fn a_dispatch_during_the_render_is_a_render_phase_update() {
        let mut harness = Harness::new();
        harness.begin();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        state.set(2);
        // The root is not asked for a render: the pass that is running replays
        // the update itself.
        assert_eq!(harness.signals(), 0);
        assert!(harness.took_update());

        // The next pass applies it, and no further pass is owed.
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        assert_eq!(*state.get(), 2);
        assert!(!harness.took_update());
        harness.end();

        // Once the component is done rendering, a dispatch is an ordinary
        // update again.
        state.set(3);
        assert_eq!(harness.signals(), 1);
    }

    #[test]
    fn a_render_phase_update_takes_the_lanes_of_the_render() {
        let mut harness = Harness::new();
        harness.begin();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        // The transition lane is not part of the render, but the update belongs
        // to the render that queued it, so it applies in the same render.
        state.set_with_lane(Lanes::TRANSITION, 2);
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        assert_eq!(*state.get(), 2);
        assert!(harness.skipped.is_empty());
        harness.end();
    }

    #[test]
    fn a_dispatch_to_another_component_is_scheduled_normally() {
        let mut harness = Harness::new();
        let mut other = Harness::new();
        other.fiber = FiberId::from_index(1);
        let foreign = other.render(|ctx| ctx.use_state(|| 1_u32));

        harness.begin();
        harness.render(|ctx| ctx.use_state(|| 1_u32));
        foreign.set(2);
        assert_eq!(other.signals(), 1);
        assert!(!harness.took_update());
        harness.end();
    }

    #[test]
    fn the_committed_alternate_of_the_rendering_fiber_owns_the_queue() {
        let mut harness = Harness::new();
        // The first render makes the harness fiber the committed one and gives
        // the queue its owner.
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));

        // A handle captured before the first hook of a render still names the
        // queue through the committed fiber, which is the alternate of the one
        // being rendered.
        let mut mailbox = harness.mailbox.borrow_mut();
        mailbox.begin_render(FiberId::from_index(9), Some(harness.fiber), harness.lanes);
        drop(mailbox);
        state.set(2);
        assert_eq!(harness.signals(), 0);
        assert!(harness.took_update());

        // The work-in-progress fiber renders the update in the next pass.
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        assert_eq!(*state.get(), 2);
        harness.end();
    }
}
