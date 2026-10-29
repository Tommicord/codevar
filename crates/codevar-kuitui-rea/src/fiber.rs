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

//! The double-buffered fiber arena.
//!
//! A fiber is a node of the tree the reconciler works on. It is stored in one
//! arena and addressed by [`FiberId`], never by reference, so a parent can
//! point at a child and vice versa without fighting the borrow checker.
//!
//! # Double buffering
//!
//! Every fiber has an *alternate*: the slot on the other tree. The committed
//! tree is what the host currently shows; a render builds a work-in-progress
//! tree beside it, reusing each fiber's alternate. Nothing the render does can
//! be observed on screen, so an interrupted render is thrown away by pointing
//! the root back at the committed tree — the alternate is already there and
//! keeps its own flags and payloads.
//!
//! [`Arena::create_work_in_progress`] is the only way to start a node for a
//! render. It copies the committed payload into the pending slot, resets the
//! per-render bookkeeping (flags, deletions) and returns the alternate. The
//! parent's reconciliation then overwrites the pending payload if the node's
//! props changed.
//!
//! # Throwing a render away
//!
//! A render can be interrupted before it commits. The tree it built never
//! reaches the host, but the arena owns the fibers it allocated and the hook
//! queues it replayed. [`Arena::begin_render`] starts recording every fiber a
//! render starts; [`Arena::abandon_render`] frees what the render allocated,
//! returns each queue to the fiber that owns it in the committed tree and puts
//! the queue back to the state the last commit left. [`Arena::end_render`]
//! stops recording once the tree is complete, before the commit runs.
//!
//! # Suspense handlers
//!
//! While the pass walks the tree it keeps a stack of the fibers that can catch
//! a suspension: a Suspense boundary pushes itself, a visible Offscreen keeps
//! the handler it inherited, and a hidden one catches its own tree. When a
//! component suspends, the pass reads the top of the stack — React's
//! `getSuspenseHandler` — and unwinds to the boundary named there. The entry a
//! boundary pushes records how many fibers the render had started, so
//! [`Arena::capture`] can hand everything the boundary started back to the
//! committed tree with [`Arena::discard_since`].
//!
//! A boundary that renders its fallback pushes again with the handler it
//! inherited, so a suspension inside the fallback behaves like a throw inside a
//! `catch` block and reaches the next boundary out; React's
//! `pushFallbackTreeSuspenseHandler` does the same. The stack lives in the
//! arena rather than in the pass because a render that yields between two units
//! of work resumes with the same stack.
//!
//! An entry also records whether everything below it is out of sight: a
//! boundary renders its primary tree hidden while it is new or showing a
//! fallback, so a fallback further down hides nothing and may be committed at
//! once. React carries the same bit on its suspense context stack
//! (`InvisibleParentSuspenseContext`) and reads it when it decides whether a
//! render should keep the previous screen.
//!
//! # Context providers
//!
//! A provider publishes a value to every component below it. The value lives
//! in the context itself, so a provider pushes it when its begin runs and pops
//! it when its complete runs — React's `pushProvider` and `popProvider` — and
//! the stack of pushes is kept in the arena for the same reason the handler
//! stack is: a render that yields keeps publishing what the fibers it has not
//! completed publish.
//!
//! An entry records the value the context held before the push and how many
//! fibers the render had started. A suspension discards a subtree, and every
//! provider the discarded fibers pushed has to give its previous value back;
//! [`Arena::discard_since`] uses the mark for that. React unwinds the same
//! stack by popping providers in `unwindInterruptedWork` and `throwException`.
//!
//! # Why the tree is not a `Vec<Fiber>`
//!
//! Fibers are both created and destroyed between commits (a conditional
//! branch, a list item removed), so the arena keeps a free list and recycles
//! slots. The two halves of an alternate are freed together: the slot of one
//! half must never be handed out while the other half still points at it.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::any::{Any, TypeId};
use core::cell::RefCell;
use core::fmt;
use core::ops::{BitAnd, BitOr, Not};

use crate::element::{AnyContext, ComponentFn, Element, HostNode, ProviderNode, Reference, SuspenseNode};
use crate::hooks::{Cleanup, ContextDeps, EffectRecord, HookList, StoreCheck};
use crate::host::{Host, Key};
use crate::lane::Lanes;
use crate::wakeable::Wakeable;

/// Bookkeeping the commit phase needs, attached to a fiber.
///
/// Flags are per-alternate. Starting a work-in-progress node resets them (see
/// [`Fiber::work_in_progress`]), and every commit phase also clears the mask it
/// applied. The second rule is what keeps a finished tree honest: a fiber can
/// be shared with the committed tree instead of cloned, and a flag left on it
/// would make a later commit redo work that already ran. The masks together
/// cover [`Flags::ALL`].
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Flags(u16);

impl Flags {
    /// Nothing to do.
    pub(crate) const NONE: Self = Self(0);

    /// The node is new in this tree and must be inserted into the host.
    pub(crate) const PLACEMENT: Self = Self(1 << 0);

    /// The node's host description changed and must be updated.
    pub(crate) const UPDATE: Self = Self(1 << 1);

    /// The node's `deletions` list must be processed before it is mutated.
    pub(crate) const CHILD_DELETION: Self = Self(1 << 2);

    /// The node's children must be removed because its content changed kind.
    pub(crate) const CONTENT_RESET: Self = Self(1 << 3);

    /// The node has a ref to detach in the mutation phase and attach in the
    /// layout phase.
    pub(crate) const REF: Self = Self(1 << 4);

    /// The node has a layout effect to run.
    pub(crate) const LAYOUT: Self = Self(1 << 5);

    /// The node has a passive effect to run.
    pub(crate) const PASSIVE: Self = Self(1 << 6);

    /// The node's effect cleanup must run before its parent is mutated.
    pub(crate) const INSERTION: Self = Self(1 << 7);

    /// The subtree below an Offscreen fiber switched between hidden and
    /// visible, so the commit must hide or show its nearest host nodes.
    pub(crate) const VISIBILITY: Self = Self(1 << 8);

    /// A Suspense boundary caught a suspension raised below it.
    ///
    /// Reconciliation raises the flag when the boundary has to render its
    /// fallback; the begin pass consumes it, which is why no commit mask
    /// contains it.
    pub(crate) const DID_CAPTURE: Self = Self(1 << 9);

    /// The node read an external store this render has to verify before it
    /// commits.
    ///
    /// This is React's `StoreConsistency`. The flag is raised by the render,
    /// not applied by a commit phase, so no mask contains it.
    pub(crate) const STORE_CONSISTENCY: Self = Self(1 << 10);

    /// Every flag.
    pub(crate) const ALL: Self = Self((1 << 11) - 1);

    /// The flags the mutation phase applies and consumes.
    ///
    /// [`Flags::REF`] is not here: the mutation phase detaches the *committed*
    /// ref, but the flag describes the finished node, whose ref the layout
    /// phase attaches. Clearing it with the rest of the mutation work would
    /// leave the new ref unattached.
    pub(crate) const MUTATION_MASK: Self = Self(
        Self::PLACEMENT.0
            | Self::UPDATE.0
            | Self::CHILD_DELETION.0
            | Self::CONTENT_RESET.0
            | Self::INSERTION.0
            | Self::VISIBILITY.0,
    );

    /// The flags the layout phase applies and consumes.
    pub(crate) const LAYOUT_MASK: Self = Self(Self::UPDATE.0 | Self::REF.0 | Self::LAYOUT.0);

    /// The flags the passive phase applies and consumes.
    pub(crate) const PASSIVE_MASK: Self = Self(Self::PASSIVE.0);

    /// Whether no flag is set.
    #[cfg(test)]
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every flag of `other` is set.
    pub(crate) const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether any flag of `other` is set.
    pub(crate) const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Adds the flags of `other`.
    pub(crate) const fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Removes the flags of `other`.
    pub(crate) const fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }
}

impl BitOr for Flags {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitAnd for Flags {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
}

impl Not for Flags {
    type Output = Self;

    fn not(self) -> Self {
        Self(!self.0 & Self::ALL.0)
    }
}

impl fmt::Debug for Flags {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Flags({:#b})", self.0)
    }
}

/// What a fiber renders.
///
/// The payload is shared with the element that produced it, so starting a
/// work-in-progress node never copies a host description or a prop bag.
pub(crate) enum Payload<H: Host> {
    /// A node with no content of its own, such as the root.
    None,
    /// The render function of a component.
    Component(Rc<ComponentFn<H>>),
    /// A host element together with the children it arranges, shared with the
    /// committed tree until the element changes.
    ///
    /// The whole description is the payload, not just the node: an element's
    /// identity is what decides whether a fiber can keep its subtree, and the
    /// children are part of that description.
    Host(Rc<HostNode<H>>),
    /// The children a fragment describes.
    Fragment(Rc<[Element<H>]>),
    /// The primary and fallback trees of a Suspense boundary.
    Suspense(Rc<SuspenseNode<H>>),
    /// A context provider and the children it publishes its value to.
    Provider(Rc<ProviderNode<H>>),
}

impl<H: Host> Clone for Payload<H> {
    fn clone(&self) -> Self {
        match self {
            Self::None => Self::None,
            Self::Component(render) => Self::Component(Rc::clone(render)),
            Self::Host(node) => Self::Host(Rc::clone(node)),
            Self::Fragment(children) => Self::Fragment(Rc::clone(children)),
            Self::Suspense(node) => Self::Suspense(Rc::clone(node)),
            Self::Provider(node) => Self::Provider(Rc::clone(node)),
        }
    }
}

impl<H: Host> Payload<H> {
    /// Whether two payloads describe the same element.
    ///
    /// Identity, not equality: a render that hands the fiber the same `Rc` is
    /// a render that described the same thing, which is the check React makes
    /// on `memoizedProps` vs `pendingProps`. Only then can the fiber keep the
    /// subtree it rendered last time.
    pub(crate) fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::None, Self::None) => true,
            (Self::Component(a), Self::Component(b)) => Rc::ptr_eq(a, b),
            (Self::Host(a), Self::Host(b)) => Rc::ptr_eq(a, b),
            (Self::Fragment(a), Self::Fragment(b)) => Rc::ptr_eq(a, b),
            (Self::Suspense(a), Self::Suspense(b)) => Rc::ptr_eq(a, b),
            (Self::Provider(a), Self::Provider(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// The reference callback of a host payload, when it has one.
    pub(crate) fn reference(&self) -> Option<&Reference<H>> {
        match self {
            Self::Host(node) => node.reference.as_ref(),
            _ => None,
        }
    }
}

/// The cleanup a reference callback returned when it was attached.
///
/// The two alternates of a fiber share the cell. React copies `refCleanup`
/// into the work-in-progress fiber, so a render that never touches the ref
/// still leaves the cleanup on the fiber that becomes current, and a detach
/// clears it on both halves so it cannot run twice. One shared cell gives Rea
/// both properties while the cleanup stays an `FnOnce`.
pub(crate) type RefCleanup = Rc<RefCell<Option<Cleanup>>>;

/// The wakeables a Suspense boundary is already listening to.
///
/// This is React's retry cache, the `PossiblyWeakSet` a boundary keeps in its
/// `stateNode`. Both alternates of the boundary share it, because a listener
/// outlives the half that attached it and a wakeable must not be subscribed
/// twice when the boundary commits again. A wakeable leaves the set when it
/// resolves: work that completed is never thrown again, so nothing needs the
/// listener any more.
pub(crate) type RetryCache = Rc<RefCell<Vec<Wakeable>>>;

/// What kind of node a fiber is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FiberTag {
    /// The root of a tree. Its `state_node` is the host container.
    Root,
    /// A component; its children are whatever the component returned.
    Component,
    /// A host element.
    Host,
    /// A transparent node around a list of children.
    Fragment,
    /// A Suspense boundary; its children are the primary tree and, while the
    /// boundary shows its fallback, the fallback beside the hidden primary.
    Suspense,
    /// The container of a Suspense boundary's primary tree.
    ///
    /// React renders the primary tree inside an Offscreen fiber so the
    /// boundary can switch to its fallback without unmounting that tree. The
    /// fiber has no host node of its own; it keeps the primary tree's fibers
    /// and node identity across the switch.
    Offscreen,
    /// A context provider.
    ///
    /// The fiber has no host node of its own: its begin publishes the value
    /// its payload carries and its complete puts the previous value back, so
    /// the subtree between them reads it. React's `ContextProvider` tag does
    /// the same in `updateContextProvider` and `completeWork`.
    Provider,
}

/// Whether the host nodes below an Offscreen fiber are shown.
///
/// A boundary that shows its fallback keeps the primary tree mounted with its
/// host nodes hidden, so the tree survives until the data arrives. React keeps
/// the same bit in the Offscreen fiber's `memoizedState`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum Visibility {
    /// The subtree is part of what the host shows.
    #[default]
    Visible,
    /// The subtree is mounted, but the host must not show it.
    Hidden,
}

/// An index into the fiber arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FiberId(u32);

impl FiberId {
    /// The index of the fiber in the arena.
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }

    /// Builds an id from an arena index.
    #[cfg(test)]
    pub(crate) const fn from_index(index: u32) -> Self {
        Self(index)
    }
}

/// A node of the tree the reconciler works on.
pub(crate) struct Fiber<H: Host> {
    /// What kind of node this is.
    pub(crate) tag: FiberTag,
    /// The identity of the node among its siblings, when it has one.
    pub(crate) key: Option<Key>,
    /// The component identity, used to decide whether a node can be reused.
    pub(crate) element_type: Option<TypeId>,
    /// The payload the current render produced.
    pub(crate) pending: Payload<H>,
    /// The payload the last commit applied.
    pub(crate) memoized: Payload<H>,
    /// The parent, or `None` for the root.
    pub(crate) parent: Option<FiberId>,
    /// The first child.
    pub(crate) first_child: Option<FiberId>,
    /// The next sibling.
    pub(crate) sibling: Option<FiberId>,
    /// The position among the siblings, as last reconciled.
    pub(crate) index: usize,
    /// The host node, shared by both alternates.
    pub(crate) state_node: Option<H::Node>,
    /// The cleanup the attached reference returned, shared by both alternates.
    pub(crate) ref_cleanup: Option<RefCleanup>,
    /// The hooks of a component.
    pub(crate) hooks: HookList,
    /// The effects this render must commit.
    pub(crate) effects: Vec<EffectRecord>,
    /// The external store reads this render has to verify before it commits.
    ///
    /// React keeps the same list on the component's update queue; the engine
    /// keeps it on the fiber because a component has no update queue of its
    /// own.
    pub(crate) store_checks: Vec<StoreCheck>,
    /// The contexts this component read during its last render.
    ///
    /// This is React's `fiber.dependencies`. A provider that publishes a new
    /// value walks the subtree it covers and marks every fiber whose list names
    /// its context, so a consumer re-renders even when the element below the
    /// provider did not change.
    pub(crate) dependencies: ContextDeps,
    /// What must be done to this node in the commit.
    pub(crate) flags: Flags,
    /// What must be done to the subtree, including this node.
    pub(crate) subtree_flags: Flags,
    /// The children that were deleted from this node and must be removed from
    /// the host before anything is inserted.
    pub(crate) deletions: Vec<FiberId>,
    /// The lane of updates that landed on this node.
    pub(crate) lanes: Lanes,
    /// The lanes of updates anywhere below this node.
    pub(crate) child_lanes: Lanes,
    /// The wakeables a suspension caught while this node rendered.
    ///
    /// This is React's `updateQueue` on a Suspense fiber: a set of the
    /// wakeables `throw` stored on the boundary. The commit subscribes a
    /// retry listener to each one and empties the set, so a later commit does
    /// not subscribe them again.
    pub(crate) retry_queue: Vec<Wakeable>,
    /// The wakeables the boundary already listens to, shared by the two
    /// alternates. Only a Suspense fiber ever carries one.
    pub(crate) retry_cache: Option<RetryCache>,
    /// Whether the host nodes below this fiber are shown.
    ///
    /// Only an Offscreen fiber ever carries [`Visibility::Hidden`]; every other
    /// tag leaves the value at its default and the commit never reads it there.
    pub(crate) visibility: Visibility,
    /// The node on the other tree.
    pub(crate) alternate: Option<FiberId>,
}

impl<H: Host> Fiber<H> {
    /// Creates an empty fiber of the given kind.
    pub(crate) fn new(tag: FiberTag) -> Self {
        Self {
            tag,
            key: None,
            element_type: None,
            pending: Payload::None,
            memoized: Payload::None,
            parent: None,
            first_child: None,
            sibling: None,
            index: 0,
            state_node: None,
            ref_cleanup: None,
            hooks: HookList::default(),
            effects: Vec::new(),
            store_checks: Vec::new(),
            dependencies: Vec::new(),
            flags: Flags::NONE,
            subtree_flags: Flags::NONE,
            deletions: Vec::new(),
            lanes: Lanes::NONE,
            child_lanes: Lanes::NONE,
            retry_queue: Vec::new(),
            retry_cache: None,
            visibility: Visibility::Visible,
            alternate: None,
        }
    }

    /// Creates the work-in-progress copy of a committed fiber.
    ///
    /// The committed payload is copied into both the pending and the memoized
    /// slot, so a node whose parent bails out keeps rendering what is on
    /// screen. The per-render bookkeeping starts empty: the commit flags, the
    /// deletions and the sibling link belong to the render that is about to
    /// start. Inheriting the old sibling would splice the committed list into
    /// the new one, so reconciliation relinks the children from scratch.
    pub(crate) fn work_in_progress(current: &Self) -> Self {
        Self {
            tag: current.tag,
            key: current.key.clone(),
            element_type: current.element_type,
            pending: current.memoized.clone(),
            memoized: current.memoized.clone(),
            parent: current.parent,
            first_child: current.first_child,
            sibling: None,
            index: current.index,
            state_node: current.state_node,
            ref_cleanup: current.ref_cleanup.clone(),
            hooks: current.hooks.clone(),
            effects: Vec::new(),
            store_checks: Vec::new(),
            // React clones the dependencies into the work-in-progress fiber
            // because the render mutates them; a fiber whose component does not
            // render keeps the committed list, which is still the one a
            // provider below has to find.
            dependencies: current.dependencies.clone(),
            flags: Flags::NONE,
            subtree_flags: Flags::NONE,
            deletions: Vec::new(),
            lanes: current.lanes,
            child_lanes: current.child_lanes,
            retry_queue: current.retry_queue.clone(),
            retry_cache: current.retry_cache.clone(),
            visibility: current.visibility,
            alternate: None,
        }
    }

    /// Records a child that must be removed from the host.
    pub(crate) fn delete_child(&mut self, child: FiberId) {
        self.flags.insert(Flags::CHILD_DELETION);
        self.deletions.push(child);
    }
}

/// A fiber a render started.
struct Started {
    /// The work-in-progress fiber.
    id: FiberId,
    /// Whether the render allocated the slot, so throwing the render away frees
    /// it. A reused alternate keeps its slot for the next render to overwrite.
    allocated: bool,
}

/// An entry of the stack that decides where a suspension lands.
///
/// A Suspense boundary pushes an entry while its primary tree renders, and so
/// does every Offscreen fiber; the *target* is the fiber a suspension is thrown
/// to, which is the boundary itself for a primary tree and whatever handler was
/// inherited for a fallback. React keeps the same pair on its handler stack:
/// the top value is what `getSuspenseHandler` returns, and the entry's owner is
/// the fiber whose `completeWork` pops it.
///
/// The entry also records how many fibers the render had started when it was
/// pushed, so a capture can hand everything the target started back to the
/// committed tree. React does that by unwinding its call stack; the arena has
/// no stack, so the offset in `started` is the boundary.
///
/// `invisible` says whether the subtree below the entry is out of sight: a
/// boundary that is new or already showing a fallback renders a primary tree
/// nobody sees, and everything it pushes inherits the mark. This is the
/// `InvisibleParentSuspenseContext` bit of React's suspense context stack.
#[derive(Clone, Copy)]
struct SuspenseHandler {
    /// The fiber that pushed the entry and pops it when it completes.
    owner: FiberId,
    /// The fiber a suspension is thrown to, or `None` when nothing in the
    /// current tree can catch one.
    target: Option<FiberId>,
    /// The number of fibers the render had started when the entry was pushed.
    mark: usize,
    /// Whether the tree rendered below the entry is out of sight.
    invisible: bool,
}

/// Where a suspension was thrown to.
///
/// Returned by [`Arena::capture`]: the boundary that takes the suspension and
/// the point in the render's `started` record from which everything belongs to
/// the subtree the boundary is about to throw away.
pub(crate) struct Captured {
    /// The boundary the suspension was thrown to.
    pub(crate) boundary: FiberId,
    /// The `started` offset the boundary was pushed at.
    pub(crate) mark: usize,
}

/// An entry of the stack of providers the render is inside of.
///
/// React sets `context._currentValue` in `pushProvider` and keeps the value it
/// replaced in the provider fiber's `memoizedState`, so `popProvider` can put
/// it back. The arena keeps both halves of that pair here, because a fiber the
/// render discards never reaches `popProvider`: the entry has to survive on
/// its own until [`Arena::unwind_providers`] hands the value back.
///
/// `mark` is the number of fibers the render had started when the entry was
/// pushed, which is what tells a discard whether the provider that pushed it
/// belongs to the subtree being thrown away.
struct ProviderFrame {
    /// The fiber that pushed the entry and pops it when it completes.
    owner: FiberId,
    /// The context the entry publishes a value for.
    context: Rc<dyn AnyContext>,
    /// The value the context held before the push.
    previous: Rc<dyn Any>,
    /// The number of fibers the render had started when the entry was pushed.
    mark: usize,
}

/// The arena that owns every fiber.
pub(crate) struct Arena<H: Host> {
    /// The slots. `None` marks a slot on the free list.
    fibers: Vec<Option<Fiber<H>>>,
    /// The indices of the free slots.
    free: Vec<u32>,
    /// The fibers the render in flight started, in the order it started them.
    started: Option<Vec<Started>>,
    /// The stack of fibers that can catch a suspension, outermost first.
    handlers: Vec<SuspenseHandler>,
    /// The stack of providers the render is inside of, outermost first.
    providers: Vec<ProviderFrame>,
}

impl<H: Host> Arena<H> {
    /// Creates an empty arena.
    pub(crate) fn new() -> Self {
        Self {
            fibers: Vec::new(),
            free: Vec::new(),
            started: None,
            handlers: Vec::new(),
            providers: Vec::new(),
        }
    }

    /// The number of live fibers.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.fibers
            .iter()
            .filter(|slot| slot.is_some())
            .count()
    }

    /// The fiber at `id`, if the slot is live.
    pub(crate) fn get(&self, id: FiberId) -> Option<&Fiber<H>> {
        self.fibers
            .get(id.index())
            .and_then(Option::as_ref)
    }

    /// The fiber at `id` mutably, if the slot is live.
    pub(crate) fn get_mut(&mut self, id: FiberId) -> Option<&mut Fiber<H>> {
        self.fibers
            .get_mut(id.index())
            .and_then(Option::as_mut)
    }

    /// Stores a fiber and returns its id, recycling a free slot when possible.
    pub(crate) fn alloc(&mut self, fiber: Fiber<H>) -> FiberId {
        let id = if let Some(index) = self.free.pop() {
            if let Some(slot) = self.fibers.get_mut(index as usize) {
                *slot = Some(fiber);
            }
            FiberId(index)
        } else {
            let index = self.fibers.len() as u32;
            self.fibers.push(Some(fiber));
            FiberId(index)
        };
        if let Some(started) = &mut self.started {
            started.push(Started { id, allocated: true });
        }
        id
    }

    /// Removes a fiber from the arena and reclaims its slot.
    ///
    /// The other half of the alternate is detached but not freed, because the
    /// caller may still need its host node to remove it from the tree.
    pub(crate) fn take(&mut self, id: FiberId) -> Option<Fiber<H>> {
        let slot = self.fibers.get_mut(id.index())?;
        let fiber = slot.take()?;
        self.free.push(id.0);
        if let Some(alternate) = fiber.alternate
            && let Some(other) = self.get_mut(alternate)
        {
            other.alternate = None;
        }
        Some(fiber)
    }

    /// Returns the alternate of `current`, or creates it.
    ///
    /// The committed fiber and its alternate share one host node; only one of
    /// them is reachable from the root at a time.
    pub(crate) fn create_work_in_progress(&mut self, current: FiberId) -> Option<FiberId> {
        let mut fiber = Fiber::work_in_progress(self.get(current)?);
        if let Some(alternate) = self
            .get(current)
            .and_then(|fiber| fiber.alternate)
        {
            fiber.alternate = Some(current);
            if let Some(slot) = self.fibers.get_mut(alternate.index())
                && slot.is_some()
            {
                *slot = Some(fiber);
                if let Some(started) = &mut self.started {
                    started.push(Started {
                        id: alternate,
                        allocated: false,
                    });
                }
                return Some(alternate);
            }
        }
        let id = self.alloc(fiber);
        if let Some(fiber) = self.get_mut(id) {
            fiber.alternate = Some(current);
        }
        if let Some(fiber) = self.get_mut(current) {
            fiber.alternate = Some(id);
        }
        Some(id)
    }

    /// Starts recording the fibers a render starts.
    ///
    /// Only a render that can be interrupted needs the record; the work loop
    /// calls this when it prepares a fresh tree.
    pub(crate) fn begin_render(&mut self) {
        debug_assert!(
            self.started.is_none(),
            "a render cannot start before the previous one ended"
        );
        debug_assert!(
            self.handlers.is_empty(),
            "a render cannot start with a handler its predecessor left behind"
        );
        debug_assert!(
            self.providers.is_empty(),
            "a render cannot start with a provider its predecessor left behind"
        );
        self.started = Some(Vec::new());
    }

    /// Stops recording the fibers a render starts.
    ///
    /// The tree is complete at this point, so it is about to become the
    /// committed one and there is nothing left to throw away.
    pub(crate) fn end_render(&mut self) {
        debug_assert!(
            self.handlers.is_empty(),
            "a finished render must pop every handler it pushed"
        );
        debug_assert!(
            self.providers.is_empty(),
            "a finished render must pop every provider it pushed"
        );
        self.started = None;
    }

    /// Throws away the tree the render in flight built.
    ///
    /// Every fiber the render started is detached from its update queues: the
    /// queues return to the fiber that owns them in the committed tree, because
    /// a state handle must not wake a fiber the render abandoned — and a freed
    /// slot is handed out again. The queue itself goes back to the state the
    /// last commit left, so the next render can replay the updates this one
    /// consumed. The fibers the render allocated are freed; a reused alternate
    /// keeps its slot, which the next render overwrites from the committed
    /// fiber.
    pub(crate) fn abandon_render(&mut self) {
        self.handlers.clear();
        self.restore_providers();
        let Some(started) = self.started.take() else {
            return;
        };
        for started in started.into_iter().rev() {
            self.discard(started);
        }
    }

    /// Throws away everything the render started from `mark` on.
    ///
    /// This is the arena's half of React's `unwindUnitOfWork`: the fibers the
    /// boundary started below the point it was pushed at are the subtree the
    /// suspension discards, and the ones started before it are what the render
    /// keeps. They are dropped in reverse order so a fiber is freed after the
    /// children that are still linked from it.
    pub(crate) fn discard_since(&mut self, mark: usize) {
        self.unwind_providers(mark);
        let Some(started) = self.started.as_mut() else {
            return;
        };
        let discarded = started.split_off(mark.min(started.len()));
        for started in discarded.into_iter().rev() {
            self.discard(started);
        }
    }

    /// Returns one fiber of the render in flight to the committed tree.
    fn discard(&mut self, started: Started) {
        let Some(fiber) = self.get(started.id) else {
            return;
        };
        fiber.hooks.restore_committed();
        match fiber.alternate {
            Some(owner) => fiber.hooks.reset_owners(owner),
            None => fiber.hooks.detach(),
        }
        if started.allocated {
            self.take(started.id);
        }
    }

    /// The number of fibers the render in flight has started.
    fn started_len(&self) -> usize {
        self.started.as_ref().map_or(0, Vec::len)
    }

    /// Pushes a fiber that can catch a suspension.
    ///
    /// `target` is the fiber a suspension is thrown to: the boundary itself
    /// while its primary tree renders, and the handler the push inherited while
    /// its fallback renders — a throw inside a fallback behaves like a throw
    /// inside a `catch` block. React keeps the two apart on its stack for the
    /// same reason: the entry's owner pops it, its value is read by the throw.
    ///
    /// The entry inherits whether the tree around it is out of sight, which is
    /// what React's `setDefaultShallowSuspenseContext` keeps when it replaces
    /// the shallow half of the context.
    pub(crate) fn push_handler(&mut self, owner: FiberId, target: Option<FiberId>) {
        let invisible = self.handler_invisible();
        self.push_handler_with(owner, target, invisible);
    }

    /// Pushes a fiber that catches suspensions in a tree nobody sees.
    ///
    /// This is React's `addSubtreeSuspenseContext(…, InvisibleParentSuspenseContext)`:
    /// a boundary that is new or already showing a fallback renders a primary
    /// tree that is not on screen, so a fallback further down hides nothing
    /// that was visible.
    pub(crate) fn push_invisible_handler(&mut self, owner: FiberId, target: Option<FiberId>) {
        self.push_handler_with(owner, target, true);
    }

    /// Pushes an entry with the visibility the caller decided on.
    fn push_handler_with(&mut self, owner: FiberId, target: Option<FiberId>, invisible: bool) {
        self.handlers.push(SuspenseHandler {
            owner,
            target,
            mark: self.started_len(),
            invisible,
        });
    }

    /// The fiber a suspension would be thrown to, if any.
    pub(crate) fn handler_target(&self) -> Option<FiberId> {
        self.handlers
            .last()
            .and_then(|handler| handler.target)
    }

    /// Whether the tree being rendered is out of sight.
    ///
    /// This is React's `hasSuspenseContext(suspenseStackCursor.current,
    /// InvisibleParentSuspenseContext)`. The mark is inherited by every entry
    /// below the one that set it, so reading the top of the stack is enough.
    pub(crate) fn handler_invisible(&self) -> bool {
        self.handlers
            .last()
            .is_some_and(|handler| handler.invisible)
    }

    /// Pops the entry `owner` pushed.
    ///
    /// Pops are matched by owner so a completion that arrives without a begin —
    /// a fiber cloned by a bailout is completed without ever being begun — does
    /// not pop an entry that belongs to someone else.
    pub(crate) fn pop_handler(&mut self, owner: FiberId) {
        if self
            .handlers
            .last()
            .is_some_and(|handler| handler.owner == owner)
        {
            self.handlers.pop();
        }
    }

    /// Pops the stack down to the boundary a suspension is thrown to.
    ///
    /// This is React's `throwException`: the top target decides which boundary
    /// catches the suspension, and every entry inside it goes away with the
    /// subtree the boundary is about to discard. The returned mark is where
    /// that subtree begins in the render's `started` record, which
    /// [`Arena::discard_since`] consumes.
    pub(crate) fn capture(&mut self) -> Option<Captured> {
        let target = self.handlers.last()?.target?;
        let index = self
            .handlers
            .iter()
            .rposition(|handler| handler.owner == target)?;
        let mark = self.handlers.get(index)?.mark;
        self.handlers.truncate(index);
        Some(Captured {
            boundary: target,
            mark,
        })
    }

    /// The number of handler entries on the stack.
    #[cfg(test)]
    pub(crate) fn handler_depth(&self) -> usize {
        self.handlers.len()
    }

    /// Publishes a provider's value to everything that renders below it.
    ///
    /// This is React's `pushProvider`: the context starts reporting `value`,
    /// and the entry keeps the value it replaced so the push can be undone.
    /// The entry records how many fibers the render has started, because a
    /// later discard has to tell this provider's subtree from the tree outside
    /// it.
    pub(crate) fn push_provider(&mut self, owner: FiberId, context: Rc<dyn AnyContext>, value: Rc<dyn Any>) {
        let previous = context.swap(value);
        self.providers.push(ProviderFrame {
            owner,
            context,
            previous,
            mark: self.started_len(),
        });
    }

    /// Puts the value a provider replaced back when the provider completes.
    ///
    /// This is React's `popProvider`. The pop is matched by owner, so a fiber
    /// that completes without having pushed — the way a bailout clones a fiber
    /// it never began — leaves the stack of some other provider alone.
    pub(crate) fn pop_provider(&mut self, owner: FiberId) {
        if self
            .providers
            .last()
            .is_some_and(|frame| frame.owner == owner)
        {
            self.pop_last_provider();
        }
    }

    /// Puts back every value a provider pushed for the subtree from `mark` on.
    ///
    /// This is the provider half of React's `unwindInterruptedWork`: the
    /// fibers a capture discards include the providers that published a value
    /// below the boundary, and those values must go back before the boundary
    /// renders again. A frame pushed before the subtree began has a mark at or
    /// below `mark` — the providers the render is still inside of — and stays.
    fn unwind_providers(&mut self, mark: usize) {
        while self
            .providers
            .last()
            .is_some_and(|frame| frame.mark > mark)
        {
            self.pop_last_provider();
        }
    }

    /// Puts back every value the render in flight published.
    fn restore_providers(&mut self) {
        while !self.providers.is_empty() {
            self.pop_last_provider();
        }
    }

    /// Restores one provider frame, innermost first.
    fn pop_last_provider(&mut self) {
        if let Some(frame) = self.providers.pop() {
            frame.context.restore(frame.previous);
        }
    }

    /// The number of provider entries on the stack.
    #[cfg(test)]
    pub(crate) fn provider_depth(&self) -> usize {
        self.providers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct TestNode(u32);

    struct TestHost {
        nodes: Vec<u32>,
    }

    impl Host for TestHost {
        type Element = u32;
        type Node = TestNode;

        fn element_matches(&self, old: &u32, new: &u32) -> bool {
            old == new
        }

        fn create(&mut self, element: &u32) -> TestNode {
            self.nodes.push(*element);
            TestNode(self.nodes.len() as u32)
        }

        fn update(&mut self, _node: TestNode, _element: &u32) {}

        fn insert_before(&mut self, _parent: TestNode, _child: TestNode, _anchor: Option<TestNode>) {}

        fn remove(&mut self, _parent: TestNode, _child: TestNode) {}
    }

    fn arena() -> Arena<TestHost> {
        Arena::new()
    }

    fn fiber(tag: FiberTag) -> Fiber<TestHost> {
        Fiber::new(tag)
    }

    fn component_payload() -> Payload<TestHost> {
        Payload::Component(Rc::new(|_| Element::empty()))
    }

    #[test]
    fn allocation_recycles_freed_slots() {
        let mut arena = arena();
        let first = arena.alloc(fiber(FiberTag::Host));
        arena.alloc(fiber(FiberTag::Host));
        assert_eq!(arena.len(), 2);
        assert!(arena.take(first).is_some());
        assert_eq!(arena.len(), 1);
        let third = arena.alloc(fiber(FiberTag::Fragment));
        assert_eq!(third, first);
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.get(third).map(|fiber| fiber.tag), Some(FiberTag::Fragment));
    }

    #[test]
    fn taking_a_fiber_detaches_its_alternate() {
        let mut arena = arena();
        let current = arena.alloc(fiber(FiberTag::Component));
        let work = arena
            .create_work_in_progress(current)
            .unwrap_or(current);
        assert_eq!(
            arena
                .get(current)
                .and_then(|fiber| fiber.alternate),
            Some(work)
        );
        assert!(arena.take(current).is_some());
        assert_eq!(arena.get(work).and_then(|fiber| fiber.alternate), None);
    }

    #[test]
    fn work_in_progress_reuses_the_alternate() {
        let mut arena = arena();
        let mut current = fiber(FiberTag::Component);
        current.element_type = Some(TypeId::of::<u32>());
        current.pending = component_payload();
        current.memoized = current.pending.clone();
        current.flags = Flags::UPDATE | Flags::PLACEMENT;
        current.deletions.push(FiberId(7));
        let current = arena.alloc(current);

        let first = arena.create_work_in_progress(current);
        let second = arena.create_work_in_progress(current);
        assert_eq!(first, second);
        assert_eq!(arena.len(), 2);

        let work = first.and_then(|id| arena.get(id));
        assert!(work.is_some());
        let Some(work) = work else { return };
        assert!(work.flags.is_empty());
        assert_eq!(work.subtree_flags, Flags::NONE);
        assert!(work.deletions.is_empty());
        assert_eq!(work.element_type, Some(TypeId::of::<u32>()));
        assert_eq!(work.alternate, Some(current));
        assert!(matches!(work.pending, Payload::Component(_)));
    }

    #[test]
    fn abandoning_a_render_frees_what_it_allocated() {
        let mut arena = arena();
        let committed = arena.alloc(fiber(FiberTag::Component));
        arena.begin_render();
        let Some(work) = arena.create_work_in_progress(committed) else {
            return;
        };
        // A fiber the render created from scratch, the way reconciliation
        // allocates a child with no committed counterpart.
        let created = arena.alloc(fiber(FiberTag::Host));
        assert_eq!(arena.len(), 3);

        arena.abandon_render();
        assert_eq!(arena.len(), 1);
        assert!(arena.get(work).is_none());
        assert!(arena.get(created).is_none());
        assert!(arena.get(committed).is_some());
        // The committed fiber no longer points at the freed half.
        assert_eq!(
            arena
                .get(committed)
                .and_then(|fiber| fiber.alternate),
            None
        );
    }

    #[test]
    fn abandoning_a_render_keeps_a_reused_alternate() {
        let mut arena = arena();
        let committed = arena.alloc(fiber(FiberTag::Component));
        let Some(alternate) = arena.create_work_in_progress(committed) else {
            return;
        };

        // The slot an earlier render allocated is reused, not allocated again.
        arena.begin_render();
        assert_eq!(arena.create_work_in_progress(committed), Some(alternate));
        arena.abandon_render();

        assert_eq!(arena.len(), 2);
        assert_eq!(
            arena
                .get(committed)
                .and_then(|fiber| fiber.alternate),
            Some(alternate)
        );
        assert_eq!(
            arena
                .get(alternate)
                .and_then(|fiber| fiber.alternate),
            Some(committed)
        );
    }

    #[test]
    fn a_completed_render_stops_recording() {
        let mut arena = arena();
        arena.alloc(fiber(FiberTag::Component));
        arena.begin_render();
        arena.end_render();
        // The tree committed: a later abandon has nothing to throw away.
        let work = arena.alloc(fiber(FiberTag::Host));
        arena.abandon_render();
        assert_eq!(arena.len(), 2);
        assert!(arena.get(work).is_some());
    }

    #[test]
    fn work_in_progress_copies_children_and_lanes() {
        let mut arena = arena();
        let mut current = fiber(FiberTag::Host);
        let child = arena.alloc(fiber(FiberTag::Host));
        current.first_child = Some(child);
        current.child_lanes = Lanes::DEFAULT;
        current.lanes = Lanes::SYNC;
        current.state_node = Some(TestNode(3));
        let current = arena.alloc(current);

        let Some(work) = arena
            .create_work_in_progress(current)
            .and_then(|id| arena.get(id))
        else {
            return;
        };
        assert_eq!(work.first_child, Some(child));
        assert_eq!(work.child_lanes, Lanes::DEFAULT);
        assert_eq!(work.lanes, Lanes::SYNC);
        assert_eq!(work.state_node, Some(TestNode(3)));
        assert!(matches!(work.pending, Payload::None));
    }

    #[test]
    fn a_work_in_progress_fiber_shares_the_ref_cleanup() {
        let mut arena = arena();
        let mut current = fiber(FiberTag::Host);
        let cleanup: RefCleanup = Rc::new(RefCell::new(Some(Box::new(|| {}))));
        current.ref_cleanup = Some(Rc::clone(&cleanup));
        current.state_node = Some(TestNode(1));
        let current = arena.alloc(current);

        let Some(work) = arena.create_work_in_progress(current) else {
            return;
        };
        let carried = arena
            .get(work)
            .and_then(|fiber| fiber.ref_cleanup.as_ref());
        // A render that never touches the ref still has to hand the cleanup to
        // the fiber that becomes current, or a later detach cannot run it.
        assert!(carried.is_some_and(|carried| Rc::ptr_eq(carried, &cleanup)));
    }

    #[test]
    fn deleting_a_child_marks_the_parent() {
        let mut fiber = fiber(FiberTag::Host);
        fiber.delete_child(FiberId(4));
        assert!(fiber.flags.contains(Flags::CHILD_DELETION));
        assert_eq!(fiber.deletions, [FiberId(4)]);
    }

    #[test]
    fn a_payload_is_shared_between_alternates() {
        let element: Rc<HostNode<TestHost>> = Rc::new(HostNode {
            element: Rc::new(7),
            children: Rc::from(Vec::<Element<TestHost>>::new()),
            reference: None,
        });
        let payload: Payload<TestHost> = Payload::Host(Rc::clone(&element));
        let clone = payload.clone();
        let (Payload::Host(a), Payload::Host(b)) = (payload, clone) else {
            return;
        };
        assert!(Rc::ptr_eq(&a, &b));
    }

    #[test]
    fn the_phase_masks_cover_every_flag_a_phase_applies() {
        let masks = Flags::MUTATION_MASK | Flags::LAYOUT_MASK | Flags::PASSIVE_MASK;
        // A flag outside every mask would survive the commit that applied it.
        // `DID_CAPTURE` and `STORE_CONSISTENCY` are the exceptions: they are
        // raised and cleared in the render, never applied by a commit phase.
        assert!(masks.contains(Flags::ALL & !(Flags::DID_CAPTURE | Flags::STORE_CONSISTENCY)));
    }

    #[test]
    fn a_work_in_progress_fiber_keeps_the_visibility_of_the_committed_one() {
        let mut arena = arena();
        let mut current = fiber(FiberTag::Offscreen);
        current.visibility = Visibility::Hidden;
        let current = arena.alloc(current);

        let Some(work) = arena
            .create_work_in_progress(current)
            .and_then(|id| arena.get(id))
        else {
            return;
        };
        // The boundary has to know the primary tree was hidden before this
        // render to decide whether showing it again is a change.
        assert_eq!(work.visibility, Visibility::Hidden);
    }

    #[test]
    fn a_capture_pops_to_the_boundary_and_reports_its_mark() {
        let mut arena = arena();
        arena.begin_render();
        let boundary = arena.alloc(fiber(FiberTag::Suspense));
        arena.push_handler(boundary, Some(boundary));
        assert_eq!(arena.handler_target(), Some(boundary));
        let child = arena.alloc(fiber(FiberTag::Component));

        let Some(captured) = arena.capture() else {
            return;
        };
        assert_eq!(captured.boundary, boundary);
        assert_eq!(captured.mark, 1);

        // Everything the boundary started below the mark is the subtree the
        // suspension throws away; the boundary itself survives.
        arena.discard_since(captured.mark);
        assert!(arena.get(boundary).is_some());
        assert!(arena.get(child).is_none());
        assert_eq!(arena.handler_depth(), 0);
        arena.end_render();
    }

    #[test]
    fn a_capture_skips_entries_that_reuse_an_outer_boundary() {
        let mut arena = arena();
        arena.begin_render();
        let outer = arena.alloc(fiber(FiberTag::Suspense));
        arena.push_handler(outer, Some(outer));
        let inner = arena.alloc(fiber(FiberTag::Suspense));
        // A boundary showing its fallback pushes the handler it inherited, so a
        // suspension inside the fallback reaches the boundary outside it.
        arena.push_handler(inner, Some(outer));
        let child = arena.alloc(fiber(FiberTag::Component));

        let Some(captured) = arena.capture() else {
            return;
        };
        assert_eq!(captured.boundary, outer);
        assert_eq!(captured.mark, 1);

        arena.discard_since(captured.mark);
        assert!(arena.get(outer).is_some());
        // The inner boundary was started below the outer handler, so it goes
        // with the subtree the outer boundary throws away.
        assert!(arena.get(inner).is_none());
        assert!(arena.get(child).is_none());
        assert_eq!(arena.handler_depth(), 0);
        arena.end_render();
    }

    /// A context whose current value the tests can read without a consumer.
    struct TestContext {
        value: RefCell<Rc<dyn Any>>,
    }

    impl TestContext {
        fn new() -> Rc<Self> {
            Rc::new(Self {
                value: RefCell::new(Rc::new(0_u16)),
            })
        }

        fn current(&self) -> u16 {
            self.value
                .borrow()
                .clone()
                .downcast::<u16>()
                .map_or(0, |value| *value)
        }
    }

    impl AnyContext for TestContext {
        fn swap(&self, value: Rc<dyn Any>) -> Rc<dyn Any> {
            self.value.replace(value)
        }

        fn restore(&self, previous: Rc<dyn Any>) {
            *self.value.borrow_mut() = previous;
        }
    }

    #[test]
    fn a_provider_push_replaces_the_value_and_the_pop_puts_it_back() {
        let mut arena = arena();
        let context = TestContext::new();
        arena.begin_render();
        let owner = arena.alloc(fiber(FiberTag::Provider));
        arena.push_provider(owner, Rc::clone(&context) as Rc<dyn AnyContext>, Rc::new(7_u16));
        assert_eq!(context.current(), 7);
        assert_eq!(arena.provider_depth(), 1);

        // A pop belongs to the fiber that pushed: a completion of some other
        // fiber must not publish the value back.
        arena.pop_provider(FiberId::from_index(9));
        assert_eq!(context.current(), 7);
        assert_eq!(arena.provider_depth(), 1);

        arena.pop_provider(owner);
        assert_eq!(context.current(), 0);
        assert_eq!(arena.provider_depth(), 0);
        arena.end_render();
    }

    #[test]
    fn abandoning_a_render_puts_every_provider_value_back() {
        let mut arena = arena();
        let context = TestContext::new();
        arena.begin_render();
        let outer = arena.alloc(fiber(FiberTag::Provider));
        arena.push_provider(outer, Rc::clone(&context) as Rc<dyn AnyContext>, Rc::new(5_u16));
        let inner = arena.alloc(fiber(FiberTag::Provider));
        arena.push_provider(inner, Rc::clone(&context) as Rc<dyn AnyContext>, Rc::new(6_u16));
        assert_eq!(context.current(), 6);

        arena.abandon_render();
        // The render never committed, so nothing it published may outlive it.
        assert_eq!(context.current(), 0);
        assert_eq!(arena.provider_depth(), 0);
    }

    #[test]
    fn a_discard_unwinds_only_the_providers_it_throws_away() {
        let mut arena = arena();
        let kept = TestContext::new();
        let discarded = TestContext::new();
        arena.begin_render();
        let owner = arena.alloc(fiber(FiberTag::Component));
        arena.push_provider(owner, Rc::clone(&kept) as Rc<dyn AnyContext>, Rc::new(1_u16));
        let boundary = arena.alloc(fiber(FiberTag::Suspense));
        arena.push_handler(boundary, Some(boundary));
        let child = arena.alloc(fiber(FiberTag::Component));
        arena.push_provider(child, Rc::clone(&discarded) as Rc<dyn AnyContext>, Rc::new(2_u16));

        let Some(captured) = arena.capture() else {
            return;
        };
        arena.discard_since(captured.mark);

        // The provider the render is still inside of keeps publishing; the one
        // that belonged to the subtree the boundary threw away gives its value
        // back.
        assert_eq!(kept.current(), 1);
        assert_eq!(discarded.current(), 0);
        assert_eq!(arena.provider_depth(), 1);
        arena.pop_provider(owner);
        arena.end_render();
    }
}
