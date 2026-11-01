//! Copyright 2026 Codevar Project
//! Licensed under the Apache License, Version 2.0 (the
//! "License"); you may not use this file except in
//! compliance with the License. You may obtain a copy of the
//! License at
//!
//!   https://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in
//! writing, software distributed under the License is
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//! CONDITIONS OF ANY KIND, either express or implied. See the
//! License for the specific language governing permissions and
//! limitations under the License.

//! The scrolling viewport, the region that reads its gestures and the
//! table the gestures are dispatched through.
//!
//! A viewport shows a part of content that is larger than the space it
//! has: it measures the content at the size the content wants, places it
//! shifted by the scroll offset, and clips it to its own box. What makes
//! it a *reactive* viewport rather than a static scroller is that the
//! offset is state — a wheel, a drag or a fling writes it, and the tree
//! that reads it moves.
//!
//! # The speed a release was thrown with
//!
//! A fling is thrown by the speed of the whole trailing window, not by
//! its last move, so [`VelocityTracker`] keeps the recent samples and fits
//! a line through them. The constants are the thresholds the recognition
//! uses: how far a press travels before it is a drag, how long the window
//! is, and when a fling has no speed left.

use codevar_kuitui_control::Timestamp;

use crate::geometry::Offset;

/// The axes a viewport scrolls along.
///
/// A bit set rather than two flags, so a viewport that scrolls both ways
/// is one value and an axis a widget cannot scroll is a bit that is not
/// set — the same shape [`PointerButtons`] uses.
///
/// [`PointerButtons`]: crate::event::PointerButtons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ScrollAxes(u8);

impl ScrollAxes {
    /// Neither axis scrolls.
    pub const NONE: ScrollAxes = ScrollAxes(0);
    /// The vertical axis scrolls.
    pub const VERTICAL: ScrollAxes = ScrollAxes(1 << 0);
    /// The horizontal axis scrolls.
    pub const HORIZONTAL: ScrollAxes = ScrollAxes(1 << 1);
    /// Both axes scroll, the way a two dimensional canvas does.
    pub const BOTH: ScrollAxes = ScrollAxes(Self::VERTICAL.0 | Self::HORIZONTAL.0);

    /// Whether every axis of `other` scrolls.
    #[must_use]
    pub const fn contains(self, other: ScrollAxes) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no axis at all scrolls.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for ScrollAxes {
    type Output = ScrollAxes;

    fn bitor(self, rhs: ScrollAxes) -> ScrollAxes {
        ScrollAxes(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for ScrollAxes {
    fn bitor_assign(&mut self, rhs: ScrollAxes) {
        self.0 |= rhs.0;
    }
}

/// How far a drag has to travel before it stops being a press.
///
/// The slop is what keeps a click from turning into a scroll the moment
/// the hand shakes, and it is in logical pixels so it scales with the
/// window the way the platform's own slop does.
pub const TOUCH_SLOP: f32 = 4.0;

/// The window of the samples a fling is measured over, in microseconds.
///
/// Only the last moves decide the speed of a release: a pointer that
/// rested for a second before it lifted was not flicked. Android's
/// velocity tracker uses the same hundred millisecond window.
pub const VELOCITY_WINDOW: u64 = 100_000;

/// The most recent samples the tracker keeps.
const MAX_SAMPLES: usize = 20;

/// Pixels per second under which a fling is over.
///
/// Below this speed the remaining distance is under a pixel per frame,
/// so the animation would keep ticking to move nothing; Compose stops a
/// fling at the same threshold.
pub const FLING_STOP_VELOCITY: f32 = 50.0;

/// The per-second friction a fling decays through.
///
/// The model is viscous drag — the exponential decay of Compose's
/// `FloatExponentialDecaySpec`, whose default friction this is: the
/// velocity falls as `e^(-k t)` and the distance it still travels is
/// `v / k`, so a fast release carries the content further without the
/// deceleration ever becoming a hard stop.
pub const FLING_FRICTION: f32 = 1.3;

/// One position of the pointer, with the time it was reported at.
#[derive(Debug, Clone, Copy)]
struct Sample {
    /// When the backend observed the move.
    time: Timestamp,
    /// Where the pointer was.
    position: Offset,
}

/// The speeds a drag was moving at when it was released.
///
/// The tracker keeps the last [`MAX_SAMPLES`] moves and fits a straight
/// line through position over time across the trailing
/// [`VELOCITY_WINDOW`]: the slope of that fit is the velocity. A fit is
/// used instead of the last interval because a single move is one frame
/// of jitter, and a fling thrown by a jerk should not be thrown away by
/// the jerk's own last sample.
///
/// An axis whose events carry no time contributes nothing: the tracker
/// cannot invent a speed for a backend that does not report one.
#[derive(Debug, Clone)]
pub struct VelocityTracker {
    /// The samples, in a ring: `next` is where the newest goes.
    samples: [Sample; MAX_SAMPLES],
    /// Where the next sample is written.
    next: usize,
    /// How many of the ring slots hold a sample.
    filled: usize,
}

impl Default for VelocityTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl VelocityTracker {
    /// A tracker that has seen no move yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            samples: [Sample {
                time: Timestamp::UNSPECIFIED,
                position: Offset::ZERO,
            }; MAX_SAMPLES],
            next: 0,
            filled: 0,
        }
    }

    /// Records the pointer at `position`, reported at `time`.
    ///
    /// A move whose time is unspecified is dropped rather than recorded:
    /// keeping it would put an unmeasurable sample in the middle of the
    /// window and skew the fit around it.
    pub fn add(&mut self, position: Offset, time: Timestamp) {
        if !time.is_specified() {
            return;
        }
        self.samples[self.next] = Sample { time, position };
        self.next = (self.next + 1) % MAX_SAMPLES;
        if self.filled < MAX_SAMPLES {
            self.filled += 1;
        }
    }

    /// The velocity of the trailing window, in pixels per second.
    ///
    /// [`Offset::ZERO`] when fewer than two timed samples are left in the
    /// window, which is what a release after a single move is.
    #[must_use]
    pub fn velocity(&self) -> Offset {
        let Some(base) = self.window_base() else {
            return Offset::ZERO;
        };
        // The least squares fit over the window. The times are relative to
        // the oldest sample kept, which keeps their squares small enough
        // that the denominator does not lose the difference it divides by;
        // one microsecond is 1e-6 of the seconds the slope is asked for,
        // so the scaling happens once, at the end.
        let mut count = 0u64;
        let mut sum_t = 0u64;
        let mut sum_tt = 0u64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_tx = 0.0f64;
        let mut sum_ty = 0.0f64;
        for index in self.indices() {
            let sample = self.samples[index];
            let micros = sample.time.micros();
            if micros < base {
                continue;
            }
            let t = micros - base;
            count += 1;
            sum_t += t;
            sum_tt += t * t;
            sum_x += f64::from(sample.position.x);
            sum_y += f64::from(sample.position.y);
            sum_tx += t as f64 * f64::from(sample.position.x);
            sum_ty += t as f64 * f64::from(sample.position.y);
        }
        if count < 2 {
            return Offset::ZERO;
        }
        let n = count as f64;
        let sum_tf = sum_t as f64;
        let denominator = n * sum_tt as f64 - sum_tf * sum_tf;
        // Two samples at the same instant, or a fit that is a vertical
        // line, has no slope to report.
        if denominator.abs() < f64::EPSILON {
            return Offset::ZERO;
        }
        let x = (n * sum_tx - sum_tf * sum_x) / denominator;
        let y = (n * sum_ty - sum_tf * sum_y) / denominator;
        // The fit is in pixels per microsecond; a fling is asked for in
        // pixels per second.
        Offset::new((x * 1_000_000.0) as f32, (y * 1_000_000.0) as f32)
    }

    /// Forgets every sample, as a press and a release do.
    pub fn reset(&mut self) {
        self.next = 0;
        self.filled = 0;
    }

    /// The time of the oldest sample inside the trailing window.
    ///
    /// Samples older than [`VELOCITY_WINDOW`] before the newest are cut,
    /// which is also what drops the samples a resting pointer left behind.
    /// `None` when the ring holds less than two samples, since a single
    /// move has no slope either.
    fn window_base(&self) -> Option<u64> {
        if self.filled < 2 {
            return None;
        }
        let newest = self.samples[(self.next + MAX_SAMPLES - 1) % MAX_SAMPLES];
        let floor = newest
            .time
            .micros()
            .saturating_sub(VELOCITY_WINDOW);
        self.indices()
            .map(|index| self.samples[index].time.micros())
            .find(|micros| *micros >= floor)
    }

    /// The ring indices from the oldest sample kept to the newest.
    fn indices(&self) -> impl Iterator<Item = usize> + use<> {
        let start = (self.next + MAX_SAMPLES - self.filled) % MAX_SAMPLES;
        (0..self.filled).map(move |step| (start + step) % MAX_SAMPLES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_axes_combine_with_the_bit_or_operator() {
        let axes = ScrollAxes::VERTICAL | ScrollAxes::HORIZONTAL;
        assert!(axes.contains(ScrollAxes::VERTICAL));
        assert!(axes.contains(ScrollAxes::BOTH));
        assert!(!ScrollAxes::VERTICAL.contains(ScrollAxes::HORIZONTAL));
        assert!(ScrollAxes::NONE.is_empty());
    }

    #[test]
    fn the_velocity_is_the_slope_of_the_trailing_window() {
        let mut tracker = VelocityTracker::new();
        // A pointer moving 10 pixels per millisecond, sampled every
        // millisecond: the fit has to report 10_000 pixels per second.
        for step in 0..10u64 {
            tracker.add(
                Offset::new(0.0, step as f32 * 10.0),
                Timestamp::from_micros(step * 1_000),
            );
        }
        let velocity = tracker.velocity();
        assert!((velocity.y - 10_000.0).abs() < 1.0);
        assert!((velocity.x).abs() < 1.0);
    }

    #[test]
    fn a_resting_pointer_is_not_a_fling() {
        let mut tracker = VelocityTracker::new();
        tracker.add(Offset::new(0.0, 0.0), Timestamp::from_micros(10_000));
        // The pointer stopped: the last sample is far outside the window
        // of the ones that measure the release.
        tracker.add(Offset::new(0.0, 4.0), Timestamp::from_micros(900_000));
        tracker.add(Offset::new(0.0, 5.0), Timestamp::from_micros(910_000));
        let velocity = tracker.velocity();
        assert!(velocity.y.abs() < 200.0, "{velocity:?} is a resting pointer");
    }

    #[test]
    fn a_move_without_a_time_is_not_a_sample() {
        let mut tracker = VelocityTracker::new();
        tracker.add(Offset::new(0.0, 10.0), Timestamp::UNSPECIFIED);
        tracker.add(Offset::new(0.0, 20.0), Timestamp::UNSPECIFIED);
        assert_eq!(tracker.velocity(), Offset::ZERO);
    }

    #[test]
    fn two_samples_at_the_same_time_have_no_slope() {
        let mut tracker = VelocityTracker::new();
        tracker.add(Offset::new(0.0, 1.0), Timestamp::from_micros(500));
        tracker.add(Offset::new(0.0, 9.0), Timestamp::from_micros(500));
        assert_eq!(tracker.velocity(), Offset::ZERO);
    }

    #[test]
    fn the_ring_keeps_the_newest_samples() {
        let mut tracker = VelocityTracker::new();
        for step in 0..(MAX_SAMPLES as u64 + 8) {
            tracker.add(
                Offset::new(step as f32, 0.0),
                Timestamp::from_micros(step * 1_000),
            );
        }
        assert_eq!(tracker.filled, MAX_SAMPLES);
        // The eight oldest moves are gone from the ring, so the window
        // starts at the sample that displaced them.
        let oldest = tracker
            .indices()
            .next()
            .map(|index| tracker.samples[index].position.x);
        assert_eq!(oldest, Some(8.0));
        // One pixel per millisecond is the slope of every sample kept.
        let velocity = tracker.velocity();
        assert!((velocity.x - 1_000.0).abs() < 1.0, "{velocity:?}");
    }
}
