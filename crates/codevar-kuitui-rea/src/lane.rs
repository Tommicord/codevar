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

//! Priority lanes.
//!
//! A lane is one bit of a [`Lanes`] mask, and its bit position is its
//! priority: bit 0 is the most urgent work, bit 24 the least urgent. The
//! highest-priority pending work is therefore the lowest set bit, which is a
//! single `and` with the two's complement. React reached the same layout after
//! trying several designs; it makes the scheduler's hot questions — *what is
//! the next lane?* and *may this work interrupt that render?* — pure integer
//! operations.
//!
//! Bits are grouped where React groups them:
//!
//! | Bits | Group | Typical source |
//! | --- | --- | --- |
//! | 0 | [`Lanes::SYNC`] | flush of a controlled input, tests |
//! | 1 | [`Lanes::INPUT_CONTINUOUS`] | scroll, pointer move |
//! | 2 | [`Lanes::DEFAULT`] | a state update from an event handler |
//! | 3 | [`Lanes::GESTURE`] | a gesture that must beat transitions |
//! | 4–13 | [`Lanes::TRANSITION_UPDATE_LANES`] | `start_transition` work |
//! | 14–17 | [`Lanes::TRANSITION_DEFERRED_LANES`] | deferred transition values |
//! | 18–21 | [`Lanes::RETRY`] | a suspended boundary retrying |
//! | 22 | [`Lanes::IDLE`] | prefetching |
//! | 23 | [`Lanes::OFFSCREEN`] | a hidden subtree |
//! | 24 | [`Lanes::DEFERRED`] | `use_deferred_value` |
//!
//! A group is *atomic*: when any lane of a group is pending, the scheduler
//! returns the whole group. Work inside one group (two transition updates, for
//! example) must be committed together, otherwise two halves of one visual
//! change could be painted in different frames.
//!
//! # Interruption
//!
//! A render is identified by the lanes it was started for. New work interrupts
//! it only when the new highest lane is *strictly* higher priority
//! ([`Lanes::is_strictly_higher_than`]); equal or lower work waits. This is the
//! rule that keeps a transition from being restarted by another transition
//! update that arrived while it was rendering.
//!
//! # Starvation
//!
//! High-priority work can starve low-priority work forever. The root therefore
//! records when each pending lane was first seen and marks it *expired* after
//! [`Lanes::expiration_delay`]; expired work renders synchronously and time
//! slicing is disabled for it.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

/// The number of priority lanes.
pub const LANE_COUNT: usize = 25;

/// A set of priority lanes.
///
/// A lane's bit position is its priority: lower is more urgent. A `Lanes` value
/// may hold several lanes (every lane of a transition group, for example), so
/// compare priority with [`Lanes::highest`] or
/// [`Lanes::is_strictly_higher_than`], not with the numeric value of the mask.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Lanes(u32);

impl Lanes {
    /// No lane. Work that carries no lane is never scheduled.
    pub const NONE: Self = Self(0);

    /// Work that must be flushed before the browser paints — or, in this
    /// toolkit, before the current event handler returns.
    pub const SYNC: Self = Self(1 << 0);

    /// Work caused by continuous input such as scrolling or pointer movement.
    pub const INPUT_CONTINUOUS: Self = Self(1 << 1);

    /// Ordinary work, such as a state update triggered by a discrete event.
    pub const DEFAULT: Self = Self(1 << 2);

    /// Work caused by a gesture that must be committed before a transition.
    pub const GESTURE: Self = Self(1 << 3);

    /// The ten lanes a transition update can be assigned to.
    ///
    /// The lanes rotate, so two transitions started in the same event share one
    /// lane while an older transition does not block a newer one.
    pub const TRANSITION_UPDATE_LANES: Self = Self(0x3ff << 4);

    /// The four lanes reserved for values deferred by a transition.
    pub const TRANSITION_DEFERRED_LANES: Self = Self(0xf << 14);

    /// Every transition lane, update and deferred.
    pub const TRANSITION: Self = Self(Self::TRANSITION_UPDATE_LANES.0 | Self::TRANSITION_DEFERRED_LANES.0);

    /// The four lanes a suspended boundary retries on.
    pub const RETRY: Self = Self(0xf << 18);

    /// Prefetching work that may be interrupted by anything else.
    pub const IDLE: Self = Self(1 << 22);

    /// Work for a subtree that is hidden.
    pub const OFFSCREEN: Self = Self(1 << 23);

    /// Work deferred by `use_deferred_value`.
    pub const DEFERRED: Self = Self(1 << 24);

    /// Every lane.
    pub const ALL: Self = Self((1 << LANE_COUNT) - 1);

    /// The lanes React treats as blocking: sync, continuous input, default and
    /// gesture work. A render that includes one of them is not time sliced.
    pub const BLOCKING: Self =
        Self(Self::SYNC.0 | Self::INPUT_CONTINUOUS.0 | Self::DEFAULT.0 | Self::GESTURE.0);

    /// Builds a lane set, dropping bits outside [`LANE_COUNT`].
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits & Self::ALL.0)
    }

    /// Builds a lane set from a single bit index.
    ///
    /// Returns `None` when `index` is not a valid lane index.
    #[must_use]
    pub const fn from_index(index: u32) -> Option<Self> {
        if index < LANE_COUNT as u32 {
            Some(Self(1 << index))
        } else {
            None
        }
    }

    /// The raw mask.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether no lane is set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The number of lanes in the set.
    #[must_use]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Adds every lane of `other`.
    pub const fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Removes every lane of `other`.
    pub const fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }

    /// Whether every lane of `other` is set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether any lane of `other` is set.
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// The union of two sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The most urgent lane of the set, or [`Lanes::NONE`] when it is empty.
    ///
    /// This is the lowest set bit, because bit position is priority.
    #[must_use]
    pub const fn highest(self) -> Self {
        Self(self.0 & self.0.wrapping_neg())
    }

    /// The single lane index of the set, or `None` when it holds zero or more
    /// than one lane.
    ///
    /// The index is what an array keyed by lane (the root's expiration times)
    /// uses.
    #[must_use]
    pub const fn index(self) -> Option<u32> {
        if self.0 != 0 && self.0 & (self.0 - 1) == 0 {
            Some(self.0.trailing_zeros())
        } else {
            None
        }
    }

    /// The atomic group that contains the most urgent lane of the set.
    ///
    /// The whole group is returned, not just the highest bit, because work
    /// inside one group must be rendered and committed together. An empty set
    /// returns [`Lanes::NONE`].
    #[must_use]
    pub const fn select_group(self) -> Self {
        let highest = self.highest().0;
        if highest == 0 {
            Self::NONE
        } else if highest == Self::SYNC.0 {
            Self::SYNC
        } else if highest == Self::INPUT_CONTINUOUS.0 {
            Self::INPUT_CONTINUOUS
        } else if highest == Self::DEFAULT.0 {
            Self::DEFAULT
        } else if highest == Self::GESTURE.0 {
            Self::GESTURE
        } else if (highest & Self::TRANSITION_UPDATE_LANES.0) != 0 {
            Self::TRANSITION_UPDATE_LANES
        } else if (highest & Self::TRANSITION_DEFERRED_LANES.0) != 0 {
            Self::TRANSITION_DEFERRED_LANES
        } else if (highest & Self::RETRY.0) != 0 {
            Self::RETRY
        } else if highest == Self::IDLE.0 {
            Self::IDLE
        } else if highest == Self::OFFSCREEN.0 {
            Self::OFFSCREEN
        } else if highest == Self::DEFERRED.0 {
            Self::DEFERRED
        } else {
            Self::NONE
        }
    }

    /// Whether the most urgent lane of `self` outranks the most urgent lane of
    /// `other`.
    ///
    /// An empty set never outranks anything, and an empty set is never
    /// outranked. Equal highest lanes do not outrank each other.
    #[must_use]
    pub const fn is_strictly_higher_than(self, other: Self) -> bool {
        if self.0 == 0 || other.0 == 0 {
            return false;
        }
        self.highest().0 < other.highest().0
    }

    /// Whether every lane of `self` is also in `other`.
    ///
    /// The reducer uses this to decide whether an update is skipped: an update
    /// whose lane is not a subset of the render's lanes is left in the queue
    /// for a later render.
    #[must_use]
    pub const fn is_subset_of(self, other: Self) -> bool {
        other.contains(self)
    }

    /// The scheduler priority of the most urgent lane.
    #[must_use]
    pub const fn scheduler_priority(self) -> SchedulerPriority {
        let group = self.select_group();
        if group.0 == Self::SYNC.0 {
            SchedulerPriority::Immediate
        } else if group.0 == Self::INPUT_CONTINUOUS.0 {
            SchedulerPriority::UserBlocking
        } else if group.0 == Self::DEFAULT.0
            || group.0 == Self::GESTURE.0
            || group.0 == Self::TRANSITION_UPDATE_LANES.0
            || group.0 == Self::TRANSITION_DEFERRED_LANES.0
        {
            SchedulerPriority::Normal
        } else if group.0 == Self::RETRY.0 {
            SchedulerPriority::Low
        } else {
            SchedulerPriority::Idle
        }
    }

    /// How long the work may wait before it expires, in milliseconds.
    ///
    /// `None` means the lane never expires. Expired work renders in one slice
    /// and is never interrupted, which is the root's protection against a
    /// stream of urgent updates starving a transition.
    #[must_use]
    pub const fn expiration_delay(self) -> Option<u64> {
        let group = self.select_group();
        if group.0 == Self::SYNC.0 || group.0 == Self::INPUT_CONTINUOUS.0 || group.0 == Self::GESTURE.0 {
            Some(250)
        } else if group.0 == Self::DEFAULT.0
            || group.0 == Self::TRANSITION_UPDATE_LANES.0
            || group.0 == Self::TRANSITION_DEFERRED_LANES.0
            || group.0 == Self::RETRY.0
        {
            Some(5_000)
        } else {
            None
        }
    }

    /// Iterates the lanes from the most urgent to the least urgent.
    #[must_use]
    pub const fn iter(self) -> LanesIter {
        LanesIter { remaining: self.0 }
    }
}

impl BitOr for Lanes {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOrAssign for Lanes {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

impl BitAnd for Lanes {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
}

impl BitAndAssign for Lanes {
    fn bitand_assign(&mut self, other: Self) {
        self.0 &= other.0;
    }
}

impl Not for Lanes {
    type Output = Self;

    fn not(self) -> Self {
        Self(!self.0 & Self::ALL.0)
    }
}

impl fmt::Debug for Lanes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Lanes({:#x})", self.0)
    }
}

/// The priority a host scheduler runs a lane at.
///
/// The variants are ordered from most to least urgent, matching the order a
/// downstream scheduler is expected to honour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SchedulerPriority {
    /// Run before yielding to the host loop.
    Immediate,
    /// Run as soon as possible, before ordinary work.
    UserBlocking,
    /// Ordinary work.
    Normal,
    /// Work that may run after ordinary work.
    Low,
    /// Work that only runs when nothing else is pending.
    Idle,
}

/// An iterator over the lanes of a [`Lanes`] mask.
///
/// The lanes come out from the most urgent to the least urgent.
#[derive(Debug, Clone)]
pub struct LanesIter {
    /// The bits not yielded yet.
    remaining: u32,
}

impl Iterator for LanesIter {
    type Item = Lanes;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let lane = self.remaining & self.remaining.wrapping_neg();
        self.remaining &= !lane;
        Some(Lanes(lane))
    }
}

impl DoubleEndedIterator for LanesIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let lane = 1 << (31 - self.remaining.leading_zeros());
        self.remaining &= !lane;
        Some(Lanes(lane))
    }
}

impl ExactSizeIterator for LanesIter {
    fn len(&self) -> usize {
        self.remaining.count_ones() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_highest_lane_is_the_lowest_set_bit() {
        let lanes = Lanes::IDLE | Lanes::DEFAULT | Lanes::SYNC;
        assert_eq!(lanes.highest(), Lanes::SYNC);
        assert_eq!((Lanes::TRANSITION | Lanes::GESTURE).highest(), Lanes::GESTURE);
        assert_eq!(Lanes::NONE.highest(), Lanes::NONE);
    }

    #[test]
    fn a_group_is_returned_as_a_whole() {
        let lane = Lanes::TRANSITION_UPDATE_LANES.highest();
        assert_eq!(lane.select_group(), Lanes::TRANSITION_UPDATE_LANES);

        let lane = Lanes::TRANSITION_DEFERRED_LANES.highest();
        assert_eq!(lane.select_group(), Lanes::TRANSITION_DEFERRED_LANES);

        let lane = Lanes::RETRY.highest();
        assert_eq!(lane.select_group(), Lanes::RETRY);

        assert_eq!(
            (Lanes::IDLE | Lanes::TRANSITION).select_group(),
            Lanes::TRANSITION_UPDATE_LANES
        );
        assert_eq!((Lanes::SYNC | Lanes::TRANSITION).select_group(), Lanes::SYNC);
        assert_eq!(Lanes::NONE.select_group(), Lanes::NONE);
    }

    #[test]
    fn transition_lanes_are_ordered_inside_their_group() {
        let updates = Lanes::TRANSITION_UPDATE_LANES;
        let deferred = Lanes::TRANSITION_DEFERRED_LANES;
        assert!(updates.is_strictly_higher_than(deferred));
        assert!(!deferred.is_strictly_higher_than(updates));
        assert_eq!(updates | deferred, Lanes::TRANSITION);
    }

    #[test]
    fn only_strictly_higher_work_interrupts() {
        assert!(Lanes::SYNC.is_strictly_higher_than(Lanes::DEFAULT));
        assert!(!Lanes::DEFAULT.is_strictly_higher_than(Lanes::SYNC));
        assert!(!Lanes::DEFAULT.is_strictly_higher_than(Lanes::DEFAULT));
        assert!(!Lanes::NONE.is_strictly_higher_than(Lanes::IDLE));
        assert!(!Lanes::IDLE.is_strictly_higher_than(Lanes::NONE));
    }

    #[test]
    fn subset_matches_the_reducer_skip_rule() {
        let rendered = Lanes::SYNC | Lanes::DEFAULT | Lanes::IDLE;
        assert!(Lanes::SYNC.is_subset_of(rendered));
        assert!((Lanes::SYNC | Lanes::IDLE).is_subset_of(rendered));
        assert!(Lanes::NONE.is_subset_of(rendered));
        assert!(!Lanes::TRANSITION.is_subset_of(rendered));
        assert!(!rendered.is_subset_of(Lanes::SYNC));
    }

    #[test]
    fn scheduler_priorities_follow_priority_order() {
        assert_eq!(Lanes::SYNC.scheduler_priority(), SchedulerPriority::Immediate);
        assert_eq!(
            Lanes::INPUT_CONTINUOUS.scheduler_priority(),
            SchedulerPriority::UserBlocking
        );
        assert_eq!(Lanes::DEFAULT.scheduler_priority(), SchedulerPriority::Normal);
        assert_eq!(Lanes::TRANSITION.scheduler_priority(), SchedulerPriority::Normal);
        assert_eq!(Lanes::RETRY.scheduler_priority(), SchedulerPriority::Low);
        assert_eq!(Lanes::IDLE.scheduler_priority(), SchedulerPriority::Idle);
        assert_eq!(Lanes::NONE.scheduler_priority(), SchedulerPriority::Idle);
        assert!(SchedulerPriority::Immediate < SchedulerPriority::Normal);
    }

    #[test]
    fn expiration_times_match_the_lane_group() {
        assert_eq!(Lanes::SYNC.expiration_delay(), Some(250));
        assert_eq!(Lanes::INPUT_CONTINUOUS.expiration_delay(), Some(250));
        assert_eq!(Lanes::GESTURE.expiration_delay(), Some(250));
        assert_eq!(Lanes::DEFAULT.expiration_delay(), Some(5_000));
        assert_eq!(Lanes::TRANSITION.expiration_delay(), Some(5_000));
        assert_eq!(Lanes::RETRY.expiration_delay(), Some(5_000));
        assert_eq!(Lanes::IDLE.expiration_delay(), None);
        assert_eq!(Lanes::OFFSCREEN.expiration_delay(), None);
        assert_eq!(Lanes::DEFERRED.expiration_delay(), None);
        assert_eq!(Lanes::NONE.expiration_delay(), None);
    }

    #[test]
    fn iteration_is_priority_ordered() {
        let lanes = Lanes::OFFSCREEN | Lanes::SYNC | Lanes::DEFAULT | Lanes::IDLE;
        let mut iter = lanes.iter();
        assert_eq!(iter.len(), 4);
        assert_eq!(iter.next(), Some(Lanes::SYNC));
        assert_eq!(iter.next(), Some(Lanes::DEFAULT));
        assert_eq!(iter.next(), Some(Lanes::IDLE));
        assert_eq!(iter.next(), Some(Lanes::OFFSCREEN));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn iteration_can_run_backwards() {
        let lanes = Lanes::SYNC | Lanes::DEFAULT | Lanes::OFFSCREEN;
        let mut iter = lanes.iter();
        assert_eq!(iter.next_back(), Some(Lanes::OFFSCREEN));
        assert_eq!(iter.next_back(), Some(Lanes::DEFAULT));
        assert_eq!(iter.next_back(), Some(Lanes::SYNC));
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn a_single_lane_has_an_index() {
        assert_eq!(Lanes::SYNC.index(), Some(0));
        assert_eq!(Lanes::DEFAULT.index(), Some(2));
        assert_eq!(Lanes::DEFERRED.index(), Some(24));
        assert_eq!((Lanes::SYNC | Lanes::DEFAULT).index(), None);
        assert_eq!(Lanes::NONE.index(), None);
    }

    #[test]
    fn a_lane_can_be_built_from_an_index() {
        assert_eq!(Lanes::from_index(0), Some(Lanes::SYNC));
        assert_eq!(Lanes::from_index(24), Some(Lanes::DEFERRED));
        assert_eq!(Lanes::from_index(25), None);
        assert_eq!(Lanes::from_index(u32::MAX), None);
    }

    #[test]
    fn unknown_bits_are_dropped() {
        assert_eq!(Lanes::from_bits(u32::MAX), Lanes::ALL);
        assert_eq!(Lanes::from_bits(0), Lanes::NONE);
        assert_eq!(Lanes::ALL.count(), LANE_COUNT as u32);
    }

    #[test]
    fn the_complement_stays_inside_the_mask() {
        assert_eq!(!Lanes::NONE, Lanes::ALL);
        assert_eq!(!Lanes::ALL, Lanes::NONE);
        assert_eq!(!Lanes::SYNC & Lanes::ALL, Lanes::ALL & !Lanes::SYNC);
    }

    #[test]
    fn inserting_and_removing_change_the_mask() {
        let mut lanes = Lanes::NONE;
        lanes.insert(Lanes::SYNC);
        lanes.insert(Lanes::DEFAULT);
        assert!(lanes.contains(Lanes::SYNC | Lanes::DEFAULT));
        lanes.remove(Lanes::SYNC);
        assert!(!lanes.intersects(Lanes::SYNC));
        assert!(lanes.intersects(Lanes::DEFAULT));
    }
}
