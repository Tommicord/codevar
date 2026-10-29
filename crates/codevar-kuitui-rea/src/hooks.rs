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
//! A reducer update stores its action as it is, and the render that replays it
//! applies the reducer *that* render was given, so a changed reducer
//! reprocesses the actions of earlier renders. The queue is shared by both
//! kinds of hook; only the hook knows how to read its own actions.
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
//! because only then is the last rendered value the base of the update. A
//! reducer dispatch has no such comparison, because an action only means
//! something to the reducer that sees it.
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

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::any::Any;
use core::cell::{Cell, RefCell};
use core::marker::PhantomData;
use core::mem;

use crate::element::Element;
use crate::fiber::{FiberId, RetryCache};
use crate::host::Host;
use crate::lane::Lanes;
use crate::wakeable::Wakeable;

/// The cleanup of an effect.
///
/// The engine runs it before the effect runs again and when the component
/// unmounts. Returning `None` from the effect's create function means there is
/// nothing to clean up.
pub type Cleanup = Box<dyn FnOnce()>;

/// The callback a store subscription installs.
///
/// The store calls it whenever its snapshot may have changed; the hook then
/// reads the snapshot and only renders when it moved.
pub type OnStoreChange = Rc<dyn Fn()>;

/// A function that subscribes to an external store.
///
/// The argument is the [`OnStoreChange`] callback to install; the returned
/// [`Cleanup`] removes the subscription.
pub type Subscribe = Rc<dyn Fn(OnStoreChange) -> Cleanup>;

/// What a [`State`] dispatch does to the value.
///
/// A reducer's dispatch stores its action as it is instead, because React
/// applies a queued action with the reducer of the render that replays it and
/// not with the one that was current when it was queued.
enum UpdateAction<T> {
    /// Replace the value.
    Set(Rc<T>),
    /// Compute the next value from the previous one.
    Apply(Rc<dyn Fn(&T) -> T>),
}

/// Applies a state hook's action to the value the replay has reached.
///
/// The queue erases the action of every update, and the hook that owns the
/// queue says how to bring it back; this is what a [`State`] hook answers. A
/// payload that is not one of its actions belongs to a queue a different hook
/// owns, which a component can only produce by breaking the rules of hooks.
fn apply_state<T: 'static>(value: &T, action: &Rc<dyn Any>) -> Option<Rc<T>> {
    match action.downcast_ref::<UpdateAction<T>>()? {
        UpdateAction::Set(next) => Some(Rc::clone(next)),
        UpdateAction::Apply(apply) => Some(Rc::new(apply(value))),
    }
}

/// One queued update of a state hook.
#[derive(Clone)]
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

/// The queue as the last commit left it, with the updates the in-flight render
/// has taken out of [`UpdateQueue::pending`] merged in.
///
/// React keeps the same thing on the committed hook: `updateReducer` moves the
/// pending updates onto `current.baseQueue` before it processes them, so a
/// render that is thrown away leaves the committed fiber able to replay every
/// update. Rea shares one queue between the alternates, so this snapshot plays
/// that role until the commit makes the new base current.
struct CommittedBase {
    /// The base queue followed by the updates that were pending when the
    /// render reached this hook.
    updates: Vec<Update>,
    /// The state the base queue applies to.
    base_state: Option<Rc<dyn Any>>,
    /// The last rendered value before the render started.
    last_rendered: Option<Rc<dyn Any>>,
}

/// The queue shared by the alternates of a fiber.
pub(crate) struct UpdateQueue {
    /// The updates queued since they were last merged into the base queue.
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
    /// What an abandoned render puts back, while one is replaying this queue.
    committed: Option<CommittedBase>,
}

impl UpdateQueue {
    /// The fiber that currently owns the queue, if it is still attached.
    pub(crate) fn owner(&self) -> Option<FiberId> {
        self.owner.get()
    }

    /// Whether the base of the next update can differ from the last rendered
    /// value.
    ///
    /// The base queue holds the updates a render skipped and the rebased
    /// copies of the ones it applied on top of them, so a non-empty base means
    /// the next render starts from somewhere else. A pending update with a
    /// lane has the same effect, and it is also the update that marks the
    /// fiber. An update that already knows it changes nothing waits with no
    /// lane at all and does not count: React queues that one precisely so the
    /// next dispatch can still compare (see [`State::current_value`]).
    fn has_lane_bearing_work(&self) -> bool {
        !self.base.is_empty()
            || self
                .pending
                .iter()
                .any(|update| !update.lane.is_empty())
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

    /// Puts the queue back to the state the last commit left.
    ///
    /// The render that replayed the queue is being thrown away, so the base
    /// queue, the state it applies to and the last rendered value return to
    /// what the committed tree holds. Every update the render had taken out of
    /// the pending queue is back in the base queue, in source order; the next
    /// render replays them and applies the ones its lanes cover.
    fn restore_committed(&mut self) {
        let Some(committed) = self.committed.take() else {
            return;
        };
        self.base = committed.updates;
        self.base_state = committed.base_state;
        self.last_rendered = committed.last_rendered;
    }
}

/// A shared handle to one update queue.
pub(crate) type QueueHandle = Rc<RefCell<UpdateQueue>>;

/// The signal a dispatch raises for the root.
pub(crate) struct Mailbox {
    /// The queues waiting for a render, with the lanes of their updates.
    ///
    /// The queue, not the fiber, is what a dispatch can name: the fiber id of
    /// the current tree changes when alternates swap, while the queue is
    /// shared by both alternates and knows which fiber owns it right now.
    updates: Vec<(QueueHandle, Lanes)>,
    /// The lanes whose suspended work has resolved since the last drain.
    ///
    /// A ping names no queue: the work it unblocks belongs to a tree that
    /// already suspended, so the root only has to learn which lanes can now
    /// make progress. React reaches the root the same way, from the listener
    /// it attached to the wakeable.
    pinged: Lanes,
    /// The retries the listeners of a Suspense boundary raised.
    ///
    /// The listener runs whenever the work completes, long after the commit
    /// that subscribed it, so it cannot reach into the root. It leaves the
    /// boundary and its retry cache here instead; the root claims a retry lane
    /// and marks the boundary when it drains the mailbox.
    retries: Vec<RetryRequest>,
    /// The component render that is in flight, when there is one.
    render_phase: Option<RenderPhase>,
    /// The transition scope that is running, when there is one.
    ///
    /// This is React's `ReactCurrentBatchConfig.transition`: a dispatch that
    /// sees it takes a transition lane instead of the lane it was given.
    transition: Option<TransitionScope>,
    /// The transition lane every transition of the current event shares.
    ///
    /// React's `currentEventTransitionLane`. The first transition dispatch of
    /// an event claims the next lane of the rotation and the rest reuse it, so
    /// the updates of one event render together. [`Mailbox::end_event`] clears
    /// it when the root starts working, which is React's
    /// `finishEventHandler`.
    event_transition_lane: Lanes,
    /// The lane the next event's first transition is assigned.
    ///
    /// React's `nextTransitionLane`. It advances once per event that starts a
    /// transition, so consecutive events render on different lanes.
    next_transition_lane: Lanes,
}

impl Default for Mailbox {
    fn default() -> Self {
        Self {
            updates: Vec::new(),
            pinged: Lanes::NONE,
            retries: Vec::new(),
            render_phase: None,
            transition: None,
            event_transition_lane: Lanes::NONE,
            next_transition_lane: Lanes::TRANSITION_FIRST,
        }
    }
}

/// The transition scope [`start_transition`] is running.
struct TransitionScope;

/// A boundary a retry listener named.
///
/// The cache is what says the boundary is still the one the listener belongs
/// to: a fiber slot is handed out again after an unmount, and a request must
/// only mark the boundary that subscribed it.
pub(crate) struct RetryRequest {
    /// The boundary fiber the listener was attached for.
    pub(crate) boundary: FiberId,
    /// The retry cache the listener was memoized in.
    pub(crate) cache: RetryCache,
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

    /// Records that the work `lanes` were waiting for resolved.
    pub(crate) fn ping(&mut self, lanes: Lanes) {
        self.pinged.insert(lanes);
    }

    /// Takes the lanes pinged since the last drain.
    pub(crate) fn take_pinged(&mut self) -> Lanes {
        mem::take(&mut self.pinged)
    }

    /// Records that the work a boundary waited for resolved.
    pub(crate) fn retry(&mut self, request: RetryRequest) {
        self.retries.push(request);
    }

    /// Takes the retries raised since the last drain.
    pub(crate) fn drain_retries(&mut self) -> Vec<RetryRequest> {
        mem::take(&mut self.retries)
    }

    /// Whether nothing is waiting for the root.
    pub(crate) fn is_empty(&self) -> bool {
        self.updates.is_empty() && self.pinged.is_empty() && self.retries.is_empty()
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

    /// The lane a dispatch asks for `lane` should carry.
    ///
    /// A dispatch inside a transition takes a transition-update lane instead of
    /// the one it asked for, the way `requestUpdateLane` consults
    /// `ReactCurrentBatchConfig.transition` before choosing a lane. The first
    /// transition dispatch of an event claims the next lane of the rotation
    /// and every later one reuses it until [`Mailbox::end_event`], so all the
    /// updates of one event render together.
    ///
    /// A dispatch with no lane keeps it: that is the eager no-op, whose queue
    /// entry exists only so the next comparison can see it did not change the
    /// value, and it must not turn into work just because a transition is
    /// running. Render-phase updates do not come through here either; they
    /// carry the lanes of the render they belong to.
    fn transition_lane(&mut self, lane: Lanes) -> Lanes {
        if self.transition.is_none() || lane.is_empty() {
            return lane;
        }
        if self.event_transition_lane.is_empty() {
            self.event_transition_lane = self.claim_next_transition_lane();
        }
        self.event_transition_lane
    }

    /// Claims the next transition lane of the rotation.
    ///
    /// This is React's `claimNextTransitionLane`. A transition dispatch takes
    /// one of these lanes for the whole event, and `use_deferred_value` claims
    /// one directly when an urgent render has to postpone a value: the deferred
    /// render is a transition of its own and must not share the lane of the
    /// event's updates.
    fn claim_next_transition_lane(&mut self) -> Lanes {
        let lane = self.next_transition_lane;
        self.next_transition_lane = lane.next_transition();
        lane
    }

    /// Ends the event whose transitions shared a lane.
    ///
    /// This is React's `finishEventHandler`, which `performConcurrentWorkOnRoot`
    /// runs before it renders: the concurrent work loop is the heuristic that
    /// says the event is over. The next transition dispatch, from a new event,
    /// starts the rotation one lane further.
    pub(crate) fn end_event(&mut self) {
        self.event_transition_lane = Lanes::NONE;
    }
}

/// A shared handle to the root's mailbox.
pub(crate) type MailboxHandle = Rc<RefCell<Mailbox>>;

/// Creates an empty mailbox.
pub(crate) fn mailbox() -> MailboxHandle {
    Rc::new(RefCell::new(Mailbox::default()))
}

/// Runs `scope` with every dispatch inside it marked as a transition.
///
/// This is React's `startTransition`: the scope does not render anything, it
/// only asks the dispatches it runs to take a transition lane. The lane is
/// claimed from the event's cache, so the scopes of one event share it, and an
/// inner scope restores the outer one when it returns, the way
/// `ReactCurrentBatchConfig.transition` is saved and restored.
///
/// The scope restores the previous transition by running to completion rather
/// than through a guard: rea does not unwind with panics, so there is no path
/// that leaves the mailbox pointing at a transition that is over.
pub(crate) fn start_transition<R>(mailbox: &MailboxHandle, scope: impl FnOnce() -> R) -> R {
    let previous = {
        let mut mailbox = mailbox.borrow_mut();
        mailbox.transition.replace(TransitionScope)
    };
    let result = scope();
    mailbox.borrow_mut().transition = previous;
    result
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

    /// The last rendered value, while an eager comparison against it is still
    /// valid.
    ///
    /// This is the queue's side of React's `fiber.lanes === NoLanes &&
    /// alternate.lanes === NoLanes`. An update the last render did not apply
    /// makes the base of the next one differ from the value this renders, and
    /// a render in flight may leave [`UpdateQueue::last_rendered`] describing
    /// a pass that is still thrown away. The eager no-op is neither: React
    /// queues it without marking the fiber exactly because it already knows
    /// the value cannot change, so it does not stop the next dispatch from
    /// comparing against the same value.
    fn current_value(&self) -> Option<Rc<T>> {
        let queue = self.queue.borrow();
        if queue.has_lane_bearing_work() || queue.committed.is_some() {
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
        enqueue_update(&self.queue, &self.mailbox, lane, Rc::new(action));
    }
}

/// Queues `action` on `lane` and wakes the root when it is work.
///
/// This is the enqueue path of every hook that schedules work through a queue.
/// A dispatch on the queue of the component that is rendering right now is a
/// render phase update: it takes the lanes of the render, so the next pass
/// applies it, and it does not wake the root, because the render that is
/// already running replays it before the commit.
fn enqueue_update(handle: &QueueHandle, mailbox: &MailboxHandle, lane: Lanes, action: Rc<dyn Any>) {
    let mut mailbox = mailbox.borrow_mut();
    let mut queue = handle.borrow_mut();
    let render_phase = if lane.is_empty() {
        // The eager comparison already knows the value cannot change: there
        // is nothing to replay, in a render phase or anywhere else.
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
        None => {
            // A transition takes its lane after the eager no-op and the
            // render phase have had their say, so it never turns a no-op
            // into scheduled work and never overrides the render that owns
            // the queue.
            let lane = mailbox.transition_lane(lane);
            (lane, !lane.is_empty() && queue.owner.get().is_some())
        }
    };
    queue.pending.push(Update { lane, action });
    drop(queue);
    if signal {
        mailbox.signal(Rc::clone(handle), lane);
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

/// The dispatch half of a [`RenderCtx::use_reducer`] state.
///
/// The handle is plain data built from the queue, so cloning it is cheap and
/// every clone feeds the same state. It carries no value: an action only
/// changes what the component renders when a render replays it.
pub struct Dispatch<A> {
    /// The queue of the reducing hook.
    queue: QueueHandle,
    /// The root's mailbox.
    mailbox: MailboxHandle,
    /// The action type, which only the type system reads.
    action: PhantomData<A>,
}

impl<A: 'static> Dispatch<A> {
    /// Queues `action` on the default lane.
    pub fn dispatch(&self, action: A) {
        self.dispatch_with_lane(Lanes::DEFAULT, action);
    }

    /// Queues `action` on `lane`.
    ///
    /// `lane` is a single lane, as in [`State::set_with_lane`]. The enqueue
    /// path is the same one a state dispatch takes, so a transition and a
    /// render phase update win over the lane here too.
    pub fn dispatch_with_lane(&self, lane: Lanes, action: A) {
        enqueue_update(&self.queue, &self.mailbox, lane, Rc::new(action));
    }
}

impl<A> Clone for Dispatch<A> {
    fn clone(&self) -> Self {
        Self {
            queue: Rc::clone(&self.queue),
            mailbox: Rc::clone(&self.mailbox),
            action: PhantomData,
        }
    }
}

/// The state one [`RenderCtx::use_sync_external_store`] hook keeps between
/// renders.
struct ExternalStore<T> {
    /// The snapshot the last render read. React's `hook.memoizedState`.
    rendered: Rc<T>,
    /// The instance the effects keep current. React's `hook.queue`.
    instance: Rc<RefCell<StoreInstance<T>>>,
}

/// What a subscription compares the store against. React's `inst`.
struct StoreInstance<T> {
    /// The snapshot the last commit saw.
    value: Rc<T>,
    /// The reader the hook was last given.
    get_snapshot: Rc<dyn Fn() -> T>,
}

/// Whether the store moved since the snapshot the instance remembers.
///
/// React's `checkIfSnapshotChanged`. The instance is borrowed only to take the
/// reader and the value: the reader is a callback the store owns, and holding
/// the borrow across it would make a re-entrant notification panic.
fn snapshot_changed<T: PartialEq>(instance: &Rc<RefCell<StoreInstance<T>>>) -> bool {
    let (value, get_snapshot) = {
        let instance = instance.borrow();
        (Rc::clone(&instance.value), Rc::clone(&instance.get_snapshot))
    };
    get_snapshot() != *value
}

/// The dependencies of a store subscription: the subscriber, by identity.
struct SubscribeDeps(Subscribe);

impl PartialEq for SubscribeDeps {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

/// The dependencies of a store instance update: the reader and the subscriber
/// by identity, and the rendered snapshot by value.
struct StoreDeps<T> {
    /// The reader this render was given.
    get_snapshot: Rc<dyn Fn() -> T>,
    /// The subscriber this render was given.
    subscribe: Subscribe,
    /// The snapshot this render read.
    value: Rc<T>,
}

impl<T: PartialEq> PartialEq for StoreDeps<T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.get_snapshot, &other.get_snapshot)
            && Rc::ptr_eq(&self.subscribe, &other.subscribe)
            && self.value == other.value
    }
}

/// The starter of a [`RenderCtx::use_transition`] pair.
///
/// A clone is cheap and every clone drives the same pending flag. The value a
/// component captures in a handler stays valid across renders, because it holds
/// the state handle and the mailbox rather than the render scope.
#[derive(Clone)]
pub struct Transition {
    /// The pending flag the starter raises and lowers.
    pending: State<bool>,
    /// The root's mailbox, which carries the transition scope.
    mailbox: MailboxHandle,
}

impl Transition {
    /// Runs `scope` as a transition and moves the pending flag around it.
    ///
    /// The flag is raised before the scope opens, on the default lane, so the
    /// blocking render that follows shows it. It is lowered inside the scope,
    /// so that update rides the transition lane and the render that clears the
    /// flag is the one that applies the scope's work. React's `startTransition`
    /// calls `setPending` the same way around
    /// `ReactCurrentBatchConfig.transition`.
    ///
    /// Nothing renders when the call returns; the host decides when, by
    /// flushing or by handing the root a slice.
    pub fn start(&self, scope: impl FnOnce()) {
        self.pending.set(true);
        start_transition(&self.mailbox, || {
            self.pending.set(false);
            scope();
        });
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
                let mut queue = queue.borrow_mut();
                queue.last_rendered = slot.value.clone();
                // The base queue the render compacted is the committed one now,
                // so there is nothing left for an abandoned render to put back.
                queue.committed = None;
            }
        }
    }

    /// The layout effects the hooks remember, whether or not they must run.
    ///
    /// The commit creates an effect and leaves its record in the slot, so the
    /// list outlives the render. A tree that reappears after being hidden
    /// creates its layout effects from here: the commit that hid it unmounted
    /// them, and no render of the hidden tree queued a replacement.
    pub(crate) fn layout_effects(&self) -> Vec<EffectRecord> {
        self.slots
            .iter()
            .filter_map(|slot| slot.effect.clone())
            .filter(|effect| effect.tag() == EffectTag::Layout)
            .collect()
    }

    /// Runs the cleanups of every layout effect the hooks remember.
    ///
    /// The records stay in the slots: the effects come back when the tree
    /// reappears, which mounts them again.
    pub(crate) fn unmount_layout_effects(&self) {
        for slot in &self.slots {
            if let Some(effect) = &slot.effect
                && effect.tag() == EffectTag::Layout
            {
                effect.unmount();
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

    /// Puts every queue back to the state the last commit left.
    ///
    /// The work pass calls this when it throws a render away: a render that
    /// did not commit may have replayed updates and consumed them from the
    /// pending queue, and the committed tree must still be able to produce the
    /// value those updates describe.
    pub(crate) fn restore_committed(&self) {
        for slot in &self.slots {
            if let Some(queue) = &slot.queue {
                queue.borrow_mut().restore_committed();
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
    /// Whether a `use_deferred_value` hook held its value back.
    ///
    /// React keeps this flag on `hook.baseState`: an urgent render returned the
    /// previous value and claimed a lane to show the new one, so the render
    /// that catches up has to count as an update even though no update marked
    /// the fiber.
    deferred: bool,
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
    /// The wakeable of the suspension this render raised, if it raised one.
    ///
    /// A component suspends by calling [`RenderCtx::suspend`]. Rust has no
    /// unwinding, so the component runs to its end and returns an element that
    /// nothing ever sees: the work pass throws the element away and carries the
    /// wakeable to the nearest boundary, the way React carries the thenable the
    /// component threw.
    suspended: Option<Wakeable>,
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
            suspended: None,
        }
    }

    /// Ends the render, dropping the slots a shorter render did not use.
    pub(crate) fn finish(&mut self) {
        self.hooks.finish(self.cursor);
    }

    /// Suspends the component until `wakeable` reports that its work completed.
    ///
    /// React's component `throw`s a thenable; Rust has no unwinding, so a
    /// component calls this and returns the value. The element is never
    /// reconciled: the work pass throws the render of this component away and
    /// carries the wakeable to the nearest Suspense boundary, exactly as React
    /// unwinds to the boundary in `throwException`. A component that resumes
    /// without suspending renders again from the hooks it committed, so
    /// everything this pass wrote to the hook slots is discarded with it.
    ///
    /// The first suspension of a pass wins. A later call in the same pass has
    /// no boundary left to reach — the pass already has one — so it returns the
    /// same empty element and keeps the wakeable it already recorded.
    pub fn suspend<H: Host>(&mut self, wakeable: Wakeable) -> Element<H> {
        if self.suspended.is_none() {
            self.suspended = Some(wakeable);
        }
        Element::empty()
    }

    /// Takes the suspension the render raised, if it raised one.
    pub(crate) fn take_suspended(&mut self) -> Option<Wakeable> {
        self.suspended.take()
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
                let (value, applied) = Self::replay(
                    &mut queue,
                    Rc::clone(current),
                    self.render_lanes,
                    self.lanes,
                    apply_state::<T>,
                );
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
            committed: None,
        }));
        slot.value = Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>);
        slot.queue = Some(Rc::clone(&queue));
        State {
            value,
            queue,
            mailbox: Rc::clone(&self.mailbox),
        }
    }

    /// Returns a value computed by `reducer` and the dispatch that feeds it.
    ///
    /// This is React's `useReducer`. The value only changes when an action is
    /// dispatched: `reducer` receives the current value and the action and
    /// returns the next one. The action is queued as it is, not as the value
    /// the reducer returned at dispatch time, so the render that replays it
    /// applies the reducer *that* render was given. A render that changes the
    /// reducer therefore reprocesses the actions of earlier renders, the way
    /// React's `updateReducer` does with `queue.lastRenderedReducer`.
    ///
    /// `init` runs once, on mount. React's `initialArg` and optional `init`
    /// become this single closure: `use_reducer(reducer, || arg)` is the plain
    /// argument, and `use_reducer(reducer, || init(arg))` is the initializer.
    ///
    /// A reducer dispatch is never compared against the current value before
    /// it is queued: React's `dispatchReducerAction` has no eager bailout, so
    /// an action that computes the same value still renders.
    pub fn use_reducer<T: 'static, A: 'static>(
        &mut self,
        reducer: impl Fn(&T, &A) -> T + 'static,
        init: impl FnOnce() -> T,
    ) -> (Rc<T>, Dispatch<A>) {
        let reducer = Rc::new(reducer);
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
                let (value, applied) = Self::replay(
                    &mut queue,
                    Rc::clone(current),
                    self.render_lanes,
                    self.lanes,
                    |value: &T, action: &Rc<dyn Any>| {
                        action
                            .downcast_ref::<A>()
                            .map(|action| Rc::new(reducer(value, action)))
                    },
                );
                self.did_update |= applied;
                value
            };
            slot.value = Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>);
            return (
                value,
                Dispatch {
                    queue,
                    mailbox: Rc::clone(&self.mailbox),
                    action: PhantomData,
                },
            );
        }

        let value = Rc::new(init());
        let queue = Rc::new(RefCell::new(UpdateQueue {
            pending: Vec::new(),
            base: Vec::new(),
            base_state: Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>),
            owner: Cell::new(Some(self.fiber)),
            last_rendered: Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>),
            committed: None,
        }));
        slot.value = Some(Rc::new(Rc::clone(&value)) as Rc<dyn Any>);
        slot.queue = Some(Rc::clone(&queue));
        (
            value,
            Dispatch {
                queue,
                mailbox: Rc::clone(&self.mailbox),
                action: PhantomData,
            },
        )
    }

    /// Returns whether a transition is pending and the starter that runs one.
    ///
    /// This is React's `useTransition`. The starter is the same value across
    /// renders, so a handler can keep it. [`Transition::start`] raises the
    /// pending flag and runs its scope with every dispatch inside it marked as
    /// a transition: the flag and the event's urgent updates render first, then
    /// the transition's own updates render and clear it.
    ///
    /// The hook consumes two slots — the pending state and the starter — the
    /// way `mountTransition` does with its state hook and its memoized `start`.
    pub fn use_transition(&mut self) -> (bool, Transition) {
        let pending = self.use_state(|| false);
        let is_pending = *pending.get();
        let slot = Self::next_slot(self.hooks, &mut self.cursor);
        if let Some(start) = slot
            .value
            .as_deref()
            .and_then(|value| value.downcast_ref::<Transition>())
        {
            return (is_pending, start.clone());
        }
        let start = Transition {
            pending,
            mailbox: Rc::clone(&self.mailbox),
        };
        slot.value = Some(Rc::new(start.clone()) as Rc<dyn Any>);
        (is_pending, start)
    }

    /// Returns `value`, or the previous one while an urgent render catches up.
    ///
    /// This is React's `useDeferredValue`. A render that covers an urgent lane
    /// (sync, continuous input or default) cannot wait, so when the value
    /// changed it keeps the one it rendered before and claims a transition lane
    /// for the render that will show the new one. The cost of rendering the new
    /// value therefore never blocks the input that caused it; a render that
    /// covers the claimed lane renders it.
    ///
    /// The hook keeps the deferred value in its own slot rather than in a state
    /// queue: only a render that receives a new `value` can move it, never a
    /// dispatch. A gesture render is not urgent by this rule — React's
    /// `includesOnlyNonUrgentLanes` counts only the three sync-update lanes —
    /// so a gesture shows the new value immediately.
    pub fn use_deferred_value<T: PartialEq + Clone + 'static>(&mut self, value: T) -> T {
        let slot = Self::next_slot(self.hooks, &mut self.cursor);
        let Some(previous) = slot
            .value
            .as_deref()
            .and_then(|value| value.downcast_ref::<Rc<T>>())
            .cloned()
        else {
            // The first render has nothing to hold back, so the value shows as
            // it is and no lane is claimed.
            slot.value = Some(Rc::new(Rc::new(value.clone())) as Rc<dyn Any>);
            return value;
        };
        if !self.render_lanes.intersects(Lanes::SYNC_UPDATE) {
            // A render with no urgent lane was started for the deferred lane an
            // urgent render claimed, so it can show the latest value. When the
            // urgent render held the value back, no update marked the fiber:
            // this pass is the one that makes the difference visible, so it
            // records itself as an update.
            if slot.deferred {
                slot.deferred = false;
                self.did_update = true;
            }
            slot.value = Some(Rc::new(Rc::new(value.clone())) as Rc<dyn Any>);
            return value;
        }
        if *previous != value {
            // The deferred render runs on a transition lane of its own, and the
            // lane on the fiber is what makes the root render it after the
            // urgent work.
            let lane = self
                .mailbox
                .borrow_mut()
                .claim_next_transition_lane();
            self.lanes.insert(lane);
            slot.deferred = true;
        }
        (*previous).clone()
    }

    /// Returns the snapshot of an external store and subscribes to it.
    ///
    /// This is React's `useSyncExternalStore`. The snapshot is read on every
    /// render, and the store wakes the component when its subscription reports
    /// a change; a notification does not render if the snapshot it reports
    /// equals the one the last commit saw. The two functions are compared by
    /// identity, the way React compares the functions a component passes, so a
    /// component that builds `subscribe` on every render re-subscribes on
    /// every commit. Hand a stable handle — for example one captured in
    /// [`RenderCtx::use_memo`] — to subscribe once.
    ///
    /// The subscription is a passive effect: it installs after the frame, and
    /// its cleanup runs when the subscriber changes or the component unmounts.
    /// A second passive effect keeps the instance the subscription compares
    /// against current, and re-checks the store because an effect or an event
    /// may have moved it between the render and the commit. A store change
    /// renders on the sync lane, as React's `forceStoreRerender` schedules it
    /// there, so the screen never waits for the change behind lower work.
    pub fn use_sync_external_store<T: PartialEq + 'static>(
        &mut self,
        subscribe: Subscribe,
        get_snapshot: Rc<dyn Fn() -> T>,
    ) -> Rc<T> {
        let next = Rc::new(get_snapshot());
        let slot = Self::next_slot(self.hooks, &mut self.cursor);
        let (rendered, instance, queue) = match slot
            .value
            .as_deref()
            .and_then(|value| value.downcast_ref::<Rc<ExternalStore<T>>>())
            .zip(slot.queue.as_ref())
        {
            Some((store, queue)) => {
                let store = Rc::clone(store);
                let queue = Rc::clone(queue);
                queue.borrow_mut().owner.set(Some(self.fiber));
                // `updateSyncExternalStore`: a render that reads a snapshot
                // other than the one the last commit recorded has performed
                // work even though no update marked the fiber.
                if store.rendered != next {
                    self.did_update = true;
                }
                (next, Rc::clone(&store.instance), queue)
            }
            None => {
                // The queue carries no updates of its own: it exists so a
                // store notification can name the fiber through it, the way
                // React's `forceStoreRerender` schedules the fiber itself.
                let queue = Rc::new(RefCell::new(UpdateQueue {
                    pending: Vec::new(),
                    base: Vec::new(),
                    base_state: None,
                    owner: Cell::new(Some(self.fiber)),
                    last_rendered: None,
                    committed: None,
                }));
                let instance = Rc::new(RefCell::new(StoreInstance {
                    value: Rc::clone(&next),
                    get_snapshot: Rc::clone(&get_snapshot),
                }));
                slot.queue = Some(Rc::clone(&queue));
                (Rc::clone(&next), instance, queue)
            }
        };
        // The effects hold the instance, not the slot's store: a later render
        // replaces the store's rendered snapshot, while the subscription must
        // keep comparing against the one instance.
        slot.value = Some(Rc::new(Rc::new(ExternalStore {
            rendered: Rc::clone(&rendered),
            instance: Rc::clone(&instance),
        })) as Rc<dyn Any>);

        // The subscription follows `subscribe` alone, so a snapshot change
        // never replaces a subscription that is still the one the component
        // passed.
        let deps = SubscribeDeps(Rc::clone(&subscribe));
        let mailbox = Rc::clone(&self.mailbox);
        self.use_effect(deps, {
            let instance = Rc::clone(&instance);
            let queue = Rc::clone(&queue);
            let mailbox = Rc::clone(&mailbox);
            let subscribe = Rc::clone(&subscribe);
            move || {
                let instance = Rc::clone(&instance);
                let queue = Rc::clone(&queue);
                let mailbox = Rc::clone(&mailbox);
                let subscribe = Rc::clone(&subscribe);
                // `subscribeToStore`: the listener checks the snapshot the
                // instance remembers, and only a real change wakes the root.
                Some(subscribe(Rc::new(move || {
                    if snapshot_changed(&instance) {
                        mailbox
                            .borrow_mut()
                            .signal(Rc::clone(&queue), Lanes::SYNC);
                    }
                }) as Rc<dyn Fn()>))
            }
        });

        // `updateStoreInstance`, pushed whenever the reader, the subscriber or
        // the rendered snapshot changed — and on mount, which
        // `use_effect` already covers with the empty committed deps.
        self.use_effect(
            StoreDeps {
                get_snapshot: Rc::clone(&get_snapshot),
                subscribe: Rc::clone(&subscribe),
                value: Rc::clone(&rendered),
            },
            {
                let instance = Rc::clone(&instance);
                let queue = Rc::clone(&queue);
                let mailbox = Rc::clone(&mailbox);
                let value = Rc::clone(&rendered);
                move || {
                    {
                        let mut instance = instance.borrow_mut();
                        instance.value = Rc::clone(&value);
                        instance.get_snapshot = Rc::clone(&get_snapshot);
                    }
                    // An event or an earlier effect may have moved the store
                    // between the render and this effect; React checks again
                    // and forces the render the stale snapshot missed.
                    if snapshot_changed(&instance) {
                        mailbox
                            .borrow_mut()
                            .signal(Rc::clone(&queue), Lanes::SYNC);
                    }
                    None
                }
            },
        );
        rendered
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
    ///
    /// While the render runs, the merged queue — the base plus everything that
    /// was pending — stays on [`UpdateQueue::committed`], so a render the root
    /// throws away puts back exactly what the committed tree had.
    ///
    /// `apply` is how the hook that owns the queue turns one erased action into
    /// the next value. The queue cannot apply it itself: a state dispatch
    /// resolved its action at dispatch time, while a reducer keeps it raw so
    /// this render's reducer sees it.
    fn replay<T: 'static>(
        queue: &mut UpdateQueue,
        current: Rc<T>,
        render_lanes: Lanes,
        skipped: &mut Lanes,
        apply: impl Fn(&T, &Rc<dyn Any>) -> Option<Rc<T>>,
    ) -> (Rc<T>, bool) {
        let mut updates = mem::take(&mut queue.base);
        let mut pending = mem::take(&mut queue.pending);
        // The first pass of a render keeps the whole queue on the base state
        // the commit left; a later pass only adds the updates the earlier pass
        // queued, so a render that is thrown away after any pass puts back
        // every update exactly once.
        match &mut queue.committed {
            Some(committed) => committed.updates.extend(pending.iter().cloned()),
            None => {
                let mut merged = updates.clone();
                merged.extend(pending.iter().cloned());
                queue.committed = Some(CommittedBase {
                    updates: merged,
                    base_state: queue.base_state.clone(),
                    last_rendered: queue.last_rendered.clone(),
                });
            }
        }
        updates.append(&mut pending);
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
            if let Some(next) = apply(&value, &update.action) {
                value = next;
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

        /// Takes the lanes of the queued signals, the way the root drains them.
        fn drained_lanes(&mut self) -> Vec<Lanes> {
            self.mailbox
                .borrow_mut()
                .drain()
                .into_iter()
                .map(|(_, lanes)| lanes)
                .collect()
        }

        fn commit(&mut self) {
            self.hooks.commit();
        }
    }

    impl Harness {
        /// Renders into a hook list that is not the committed one, the way the
        /// work pass renders into a fiber's alternate.
        fn render_away<R>(
            harness: &Self,
            hooks: &mut HookList,
            content: impl FnOnce(&mut RenderCtx<'_>) -> R,
        ) -> R {
            harness.mailbox.borrow_mut().drain();
            let mut effects = Vec::new();
            let mut skipped = Lanes::NONE;
            let mut ids = harness.ids;
            hooks.begin_render();
            let mut ctx = RenderCtx::new(
                harness.fiber,
                hooks,
                &mut effects,
                harness.lanes,
                &mut skipped,
                Rc::clone(&harness.mailbox),
                &mut ids,
            );
            let result = content(&mut ctx);
            ctx.finish();
            result
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
    fn an_equal_update_keeps_the_eager_comparison_available() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        harness.commit();
        state.set(1);
        assert_eq!(harness.signals(), 0);
        // The update that changed nothing waits in the queue without a lane,
        // the way React keeps it in the interleaved list. It did not change
        // the value, so the next dispatch still knows what the committed one
        // is and can compare against it instead of scheduling a render.
        state.set(1);
        assert_eq!(harness.signals(), 0);
    }

    #[test]
    fn a_change_after_an_equal_update_is_still_scheduled() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        harness.commit();
        state.set(1);
        assert_eq!(harness.signals(), 0);
        // The comparison the no-op left available is against the committed
        // value, so a value that differs from it still takes the usual path.
        state.set(2);
        assert_eq!(harness.signals(), 1);
        let state = harness.render(|ctx| ctx.use_state(|| 0_u32));
        assert_eq!(*state.get(), 2);
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
    fn a_render_that_is_thrown_away_leaves_its_updates_in_the_queue() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        harness.commit();
        state.set(2);

        // A work-in-progress render replays the update and takes it out of the
        // pending queue...
        let mut thrown_away = harness.hooks.clone();
        let work = Harness::render_away(&harness, &mut thrown_away, |ctx| ctx.use_state(|| 0_u32));
        assert_eq!(*work.get(), 2);

        // ...and the render is abandoned. The committed list still holds the
        // update, so the next render produces the value it describes.
        thrown_away.restore_committed();
        let state = harness.render(|ctx| ctx.use_state(|| 0_u32));
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
    fn a_transition_starter_raises_the_flag_and_runs_the_scope() {
        let mut harness = Harness::new();
        let (pending, start, value) = harness.render(|ctx| {
            let (pending, start) = ctx.use_transition();
            let value = ctx.use_state(|| 1_u32);
            (pending, start, value)
        });
        assert!(!pending);
        harness.commit();

        start.start(|| value.set(2));
        // The flag goes up on the default lane; its lowering and the value
        // then ride the transition lane.
        assert_eq!(
            harness.drained_lanes(),
            [Lanes::DEFAULT, Lanes::TRANSITION_FIRST, Lanes::TRANSITION_FIRST]
        );

        // The default render raises the flag and leaves the transition's
        // updates pending.
        harness.lanes = Lanes::DEFAULT;
        let (pending, _, value) = harness.render(|ctx| {
            let (pending, start) = ctx.use_transition();
            let value = ctx.use_state(|| 1_u32);
            (pending, start, value)
        });
        assert!(pending);
        assert_eq!(*value.get(), 1);
        harness.commit();

        // The transition render applies them and clears the flag.
        harness.lanes = Lanes::TRANSITION_FIRST;
        let (pending, value) = harness.render(|ctx| {
            let (pending, _) = ctx.use_transition();
            (pending, ctx.use_state(|| 1_u32))
        });
        assert!(!pending);
        assert_eq!(*value.get(), 2);
    }

    #[test]
    fn a_deferred_value_lags_an_urgent_render_and_catches_up() {
        let mut harness = Harness::new();
        let shown = harness.render(|ctx| ctx.use_deferred_value(1_u32));
        assert_eq!(shown, 1);
        harness.commit();

        // The urgent render keeps the value it had and claims the first
        // transition lane, which the fiber records for the root.
        let shown = harness.render(|ctx| ctx.use_deferred_value(2_u32));
        assert_eq!(shown, 1);
        assert_eq!(harness.skipped, Lanes::TRANSITION_FIRST);

        // The render that covers the claimed lane shows the latest value.
        harness.lanes = Lanes::TRANSITION_FIRST;
        let shown = harness.render(|ctx| ctx.use_deferred_value(2_u32));
        assert_eq!(shown, 2);
        assert!(harness.skipped.is_empty());
    }

    #[test]
    fn an_unchanged_deferred_value_claims_no_lane() {
        let mut harness = Harness::new();
        harness.render(|ctx| ctx.use_deferred_value(1_u32));
        harness.commit();
        let shown = harness.render(|ctx| ctx.use_deferred_value(1_u32));
        assert_eq!(shown, 1);
        assert!(harness.skipped.is_empty());
    }

    #[test]
    fn a_gesture_render_shows_the_new_deferred_value() {
        let mut harness = Harness::new();
        harness.render(|ctx| ctx.use_deferred_value(1_u32));
        harness.commit();
        // A gesture is not in React's urgent set, so there is nothing to defer
        // and no lane is claimed.
        harness.lanes = Lanes::GESTURE;
        let shown = harness.render(|ctx| ctx.use_deferred_value(2_u32));
        assert_eq!(shown, 2);
        assert!(harness.skipped.is_empty());
    }

    #[test]
    fn a_reducer_applies_the_actions_in_order() {
        let mut harness = Harness::new();
        let reducer = |sum: &u32, action: &u32| sum + action;
        let (value, dispatch) = harness.render(|ctx| ctx.use_reducer(reducer, || 0_u32));
        assert_eq!(*value, 0);

        dispatch.dispatch(2);
        dispatch.dispatch(3);
        assert_eq!(harness.drained_lanes(), [Lanes::DEFAULT, Lanes::DEFAULT]);

        let (value, _) = harness.render(|ctx| ctx.use_reducer(reducer, || 0_u32));
        assert_eq!(*value, 5);
    }

    #[test]
    fn a_reducer_dispatch_that_recomputes_the_same_value_is_still_scheduled() {
        let mut harness = Harness::new();
        let reducer = |value: &u32, _: &()| *value;
        let (_, dispatch) = harness.render(|ctx| ctx.use_reducer(reducer, || 7_u32));

        dispatch.dispatch(());
        // React's `dispatchReducerAction` never computes an eager state, so
        // this is real work even though the reducer returns the same value.
        assert_eq!(harness.drained_lanes(), [Lanes::DEFAULT]);

        let (value, _) = harness.render(|ctx| ctx.use_reducer(reducer, || 7_u32));
        assert_eq!(*value, 7);
    }

    #[test]
    fn a_skipped_reducer_action_is_replayed_with_the_latest_reducer() {
        let add = |sum: &u32, action: &u32| sum + action;
        let multiply = |sum: &u32, action: &u32| sum * action;
        let mut harness = Harness::new();
        let (_, dispatch) = harness.render(|ctx| ctx.use_reducer(add, || 1_u32));
        dispatch.dispatch_with_lane(Lanes::DEFAULT, 3);

        // A sync render does not cover the default-lane action: the value
        // stays, and the lane is recorded so the root renders it again.
        harness.lanes = Lanes::SYNC;
        let (value, _) = harness.render(|ctx| ctx.use_reducer(add, || 1_u32));
        assert_eq!(*value, 1);
        assert_eq!(harness.skipped, Lanes::DEFAULT);

        // The next render covers the lane but passes a different reducer:
        // React replays the queued action with the reducer this render was
        // given, not the one that was current at dispatch time.
        harness.lanes = Lanes::DEFAULT;
        let (value, _) = harness.render(|ctx| ctx.use_reducer(multiply, || 1_u32));
        assert_eq!(*value, 3);
    }

    #[test]
    fn a_reducer_dispatch_while_rendering_applies_in_the_same_pass() {
        let reducer = |sum: &u32, action: &u32| sum + action;
        let mut harness = Harness::new();
        let (_, dispatch) = harness.render(|ctx| ctx.use_reducer(reducer, || 0_u32));
        harness.commit();

        harness.begin();
        dispatch.dispatch(1);
        assert!(harness.took_update());
        // The render that is already running replays it, so the root is not
        // woken and no signal is queued.
        assert_eq!(harness.signals(), 0);
        harness.end();

        let (value, _) = harness.render(|ctx| ctx.use_reducer(reducer, || 0_u32));
        assert_eq!(*value, 1);
    }

    #[test]
    fn a_reducer_dispatch_inside_a_transition_takes_a_transition_lane() {
        let reducer = |sum: &u32, action: &u32| sum + action;
        let mut harness = Harness::new();
        let (_, dispatch) = harness.render(|ctx| ctx.use_reducer(reducer, || 0_u32));
        harness.commit();

        start_transition(&harness.mailbox, || dispatch.dispatch(2));
        assert_eq!(harness.drained_lanes(), [Lanes::TRANSITION_FIRST]);
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
    fn a_dispatch_inside_a_transition_takes_a_transition_lane() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        harness.commit();
        start_transition(&harness.mailbox, || state.set(2));
        // The default lane the dispatch asked for is replaced by the lane the
        // transition was assigned.
        assert_eq!(harness.drained_lanes(), [Lanes::TRANSITION_FIRST]);

        harness.lanes = Lanes::TRANSITION_FIRST;
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        assert_eq!(*state.get(), 2);
    }

    #[test]
    fn transitions_of_one_event_share_a_lane() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        harness.commit();
        start_transition(&harness.mailbox, || state.set(2));
        start_transition(&harness.mailbox, || state.set(3));
        assert_eq!(
            harness.drained_lanes(),
            [Lanes::TRANSITION_FIRST, Lanes::TRANSITION_FIRST]
        );

        // The root clears the cache when it starts working, so the next event
        // starts one lane further into the rotation.
        harness.mailbox.borrow_mut().end_event();
        start_transition(&harness.mailbox, || state.set(4));
        assert_eq!(harness.drained_lanes(), [Lanes::from_bits(1 << 5)]);
    }

    #[test]
    fn a_no_op_inside_a_transition_claims_no_lane() {
        let mut harness = Harness::new();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        harness.commit();
        start_transition(&harness.mailbox, || state.set(1));
        // The eager comparison queued the equal update without a lane, so it
        // neither wakes the root nor takes a transition lane.
        assert_eq!(harness.signals(), 0);

        start_transition(&harness.mailbox, || state.set(2));
        assert_eq!(harness.drained_lanes(), [Lanes::TRANSITION_FIRST]);
    }

    #[test]
    fn a_transition_does_not_capture_a_render_phase_update() {
        let mut harness = Harness::new();
        harness.begin();
        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        start_transition(&harness.mailbox, || state.set(2));
        // The update belongs to the render that is running, so it keeps that
        // render's lanes and never reaches the root.
        assert_eq!(harness.signals(), 0);
        assert!(harness.took_update());

        let state = harness.render(|ctx| ctx.use_state(|| 1_u32));
        assert_eq!(*state.get(), 2);
        assert!(harness.skipped.is_empty());
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

    /// A store a component can read and a test can move, with the listener the
    /// subscription installs.
    struct TestStore {
        value: Rc<Cell<u32>>,
        listener: Rc<RefCell<Option<OnStoreChange>>>,
        subscribed: Rc<Cell<u32>>,
        unsubscribed: Rc<Cell<u32>>,
        subscribe: Subscribe,
        get_snapshot: Rc<dyn Fn() -> u32>,
    }

    impl TestStore {
        fn new(value: u32) -> Self {
            let value = Rc::new(Cell::new(value));
            let listener: Rc<RefCell<Option<OnStoreChange>>> = Rc::new(RefCell::new(None));
            let subscribed = Rc::new(Cell::new(0));
            let unsubscribed = Rc::new(Cell::new(0));
            let subscribe = {
                let current = Rc::clone(&listener);
                let subscribed = Rc::clone(&subscribed);
                let unsubscribed = Rc::clone(&unsubscribed);
                Rc::new(move |on_change: Rc<dyn Fn()>| -> Cleanup {
                    *current.borrow_mut() = Some(on_change);
                    subscribed.set(subscribed.get() + 1);
                    let current = Rc::clone(&current);
                    let unsubscribed = Rc::clone(&unsubscribed);
                    Box::new(move || {
                        *current.borrow_mut() = None;
                        unsubscribed.set(unsubscribed.get() + 1);
                    })
                }) as Subscribe
            };
            let get_snapshot = {
                let value = Rc::clone(&value);
                Rc::new(move || value.get()) as Rc<dyn Fn() -> u32>
            };
            Self {
                value,
                listener,
                subscribed,
                unsubscribed,
                subscribe,
                get_snapshot,
            }
        }

        /// Moves the store without telling anyone.
        fn set(&self, value: u32) {
            self.value.set(value);
        }

        /// Calls the listener the current subscription installed.
        fn notify(&self) {
            let listener = self.listener.borrow().clone();
            if let Some(listener) = listener {
                listener();
            }
        }

        /// A subscriber with its own identity that does the same thing, to
        /// change the function a render passes without changing the store.
        fn resubscribe(&self) -> Subscribe {
            let subscribe = Rc::clone(&self.subscribe);
            Rc::new(move |on_change| subscribe(on_change))
        }
    }

    #[test]
    fn a_store_reads_its_snapshot_and_queues_its_subscription() {
        let mut harness = Harness::new();
        let store = TestStore::new(1);
        let shown = harness.render(|ctx| {
            ctx.use_sync_external_store(Rc::clone(&store.subscribe), Rc::clone(&store.get_snapshot))
        });
        assert_eq!(*shown, 1);

        // The subscription and the instance update are passive effects, in the
        // order React pushes them. Neither has installed anything yet.
        assert_eq!(harness.effects.len(), 2);
        assert_eq!(harness.effects[0].tag(), EffectTag::Passive);
        assert_eq!(harness.effects[1].tag(), EffectTag::Passive);
        assert_eq!(store.subscribed.get(), 0);

        harness.effects[0].create();
        harness.effects[1].create();
        assert_eq!(store.subscribed.get(), 1);
        // The instance effect found the snapshot the render read, so it does
        // not wake the root.
        assert_eq!(harness.signals(), 0);
    }

    #[test]
    fn a_store_notification_wakes_only_on_a_changed_snapshot() {
        let mut harness = Harness::new();
        let store = TestStore::new(1);
        harness.render(|ctx| {
            ctx.use_sync_external_store(Rc::clone(&store.subscribe), Rc::clone(&store.get_snapshot))
        });
        harness.effects[0].create();
        harness.effects[1].create();

        // A notification that reports the snapshot the commit already saw is
        // not work.
        store.notify();
        assert_eq!(harness.signals(), 0);

        store.set(2);
        store.notify();
        // `forceStoreRerender` schedules on the sync lane, so the change never
        // waits behind lower work.
        assert_eq!(harness.drained_lanes(), [Lanes::SYNC]);

        harness.lanes = Lanes::SYNC;
        let shown = harness.render(|ctx| {
            ctx.use_sync_external_store(Rc::clone(&store.subscribe), Rc::clone(&store.get_snapshot))
        });
        assert_eq!(*shown, 2);
        // The reader and the subscriber are the same, so only the instance
        // update follows the changed snapshot.
        assert_eq!(harness.effects.len(), 1);
        harness.effects[0].create();
        assert_eq!(harness.signals(), 0);

        // The listener the first subscription installed compares against the
        // same instance this commit refreshed, so the repeated notification is
        // still nothing.
        store.notify();
        assert_eq!(harness.signals(), 0);
    }

    #[test]
    fn a_store_that_moved_before_its_effect_renders_again() {
        let mut harness = Harness::new();
        let store = TestStore::new(1);
        harness.render(|ctx| {
            ctx.use_sync_external_store(Rc::clone(&store.subscribe), Rc::clone(&store.get_snapshot))
        });

        // The store moves between the render and the passive effects, the way
        // an event or a layout effect moves it in React.
        store.set(5);
        harness.effects[0].create();
        harness.effects[1].create();

        // `updateStoreInstance` checks the store again and forces the render
        // the stale snapshot missed.
        assert_eq!(harness.drained_lanes(), [Lanes::SYNC]);
    }

    #[test]
    fn a_stable_subscription_survives_a_render() {
        let mut harness = Harness::new();
        let store = TestStore::new(1);
        harness.render(|ctx| {
            ctx.use_sync_external_store(Rc::clone(&store.subscribe), Rc::clone(&store.get_snapshot))
        });
        harness.effects[0].create();
        harness.effects[1].create();
        harness.commit();

        // Same subscriber and reader by identity, same snapshot: React pushes
        // neither effect again.
        harness.render(|ctx| {
            ctx.use_sync_external_store(Rc::clone(&store.subscribe), Rc::clone(&store.get_snapshot))
        });
        assert!(harness.effects.is_empty());
        assert_eq!(store.subscribed.get(), 1);
    }

    #[test]
    fn a_new_subscription_replaces_the_previous_one() {
        let mut harness = Harness::new();
        let store = TestStore::new(1);
        harness.render(|ctx| {
            ctx.use_sync_external_store(Rc::clone(&store.subscribe), Rc::clone(&store.get_snapshot))
        });
        harness.effects[0].create();
        harness.effects[1].create();
        harness.commit();

        let subscribe = store.resubscribe();
        harness
            .render(|ctx| ctx.use_sync_external_store(Rc::clone(&subscribe), Rc::clone(&store.get_snapshot)));
        // The subscriber changed by identity, so both effects run again.
        assert_eq!(harness.effects.len(), 2);

        // The commit cleans the old subscription up before creating the new
        // one, the way the passive unmount pass runs first.
        harness.effects[0].unmount();
        harness.effects[1].unmount();
        harness.effects[0].create();
        harness.effects[1].create();
        assert_eq!(store.unsubscribed.get(), 1);
        assert_eq!(store.subscribed.get(), 2);
        assert_eq!(harness.signals(), 0);
    }
}
