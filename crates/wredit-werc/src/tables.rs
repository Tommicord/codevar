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
//! distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
//! OR CONDITIONS OF ANY KIND, either express or implied. See
//! the License for the specific language governing
//! permissions and limitations under the License.

//! The fixed tables the coding loop reads on every bit.
//!
//! Three widths are in play. The coder codes 12-bit probabilities
//! (`1..=4095`), the mixer accumulates logits as `i32`, and the state map
//! keeps 16-bit probabilities. [`squash`] and [`stretch`] move between the
//! first two, [`clamp12`], [`clamp16`] and [`clamplog`] keep each inside the
//! domain the other one's table was built for.
//!
//! Both tables are generated data - the `SQ` and `ST` arrays of
//! `tables_data` - because a build-time computation would make a stream
//! depend on the rounding of the host libm. They are public because they
//! *are* the format: a stream is only decodable by a model that reads the
//! same squash curve.
//!
//! The bit-history state machine adds three more tables, each built by a
//! `const fn`: [`NX`] is the transition a context slot steps through,
//! [`INVR`] the learning rate a state-map visit counter asks for, and [`SM0`]
//! the entry a slot gets on first use. Those formulas are ratios of integers
//! once their rounding is written as a floor over a doubled numerator, so
//! they need no libm and no generated data - and no float appears in the
//! coding loop at all.

use crate::tables_data::{SQ, ST};

/// Lowest logit the squash table holds.
pub const LOGIT_MIN: i32 = -2047;

/// Highest logit the squash table holds.
pub const LOGIT_MAX: i32 = 2047;

/// Keep a probability inside the 12-bit domain the coder codes with.
#[inline]
pub const fn clamp12(p: i32) -> i32 {
    if p < 1 {
        1
    } else if p > 4095 {
        4095
    } else {
        p
    }
}

/// Keep a logit inside the domain [`squash`] can represent.
#[inline]
pub const fn clamplog(x: i32) -> i32 {
    if x < LOGIT_MIN {
        LOGIT_MIN
    } else if x > LOGIT_MAX {
        LOGIT_MAX
    } else {
        x
    }
}

/// The 12-bit probability of a logit.
///
/// Saturating outside `-2047..=2047` is what lets a mixer whose dot product
/// ran away still hand the coder a probability it can code with.
#[inline]
pub fn squash(x: i32) -> i32 {
    i32::from(SQ[(clamplog(x) - LOGIT_MIN) as usize])
}

/// The logit of a 12-bit probability.
///
/// Not the exact inverse of [`squash`]: a 12-bit probability cannot carry a
/// logit past about +-384, so the tails round to a smaller magnitude. The
/// mixer only needs the pair to be monotone in both directions.
#[inline]
pub fn stretch(p: i32) -> i32 {
    i32::from(ST[clamp12(p) as usize])
}

/// Keep a 16-bit state-map probability inside the domain it was built for.
#[inline]
pub const fn clamp16(p: i32) -> i32 {
    if p < 1 {
        1
    } else if p > 65534 {
        65534
    } else {
        p
    }
}

/// Bit-history transition: `(state << 1) | observed bit -> next state`.
///
/// A state is `(n0 << 5) | (n1 << 2) | last`: how many zeros and ones the
/// slot has seen, saturating at 7 each, and the bit it last saw. The other
/// count decays only once it is above 3 *and* the two sum past 7. The
/// published state machines decay as soon as the counts sum past 3, which
/// measured 7.8% more output at 1 MB: it pins every state near `(2, 2)` and
/// throws away the count magnitude that says how far to trust the frequency.
pub const NX: [u8; 512] = build_nx();

const fn build_nx() -> [u8; 512] {
    let mut table = [0u8; 512];
    let mut state = 0usize;
    while state < 256 {
        let n0 = (state >> 5) & 7;
        let n1 = (state >> 2) & 7;
        // The observed bit grows its own count; the other one may decay.
        let zeros = n0;
        let ones = if n1 < 7 { n1 + 1 } else { 7 };
        let zeros = if zeros > 3 && zeros + ones > 7 {
            zeros - 1
        } else {
            zeros
        };
        table[(state << 1) | 1] = ((zeros << 5) | (ones << 2) | 1) as u8;
        let zeros = if n0 < 7 { n0 + 1 } else { 7 };
        let ones = n1;
        let ones = if ones > 3 && zeros + ones > 7 {
            ones - 1
        } else {
            ones
        };
        table[state << 1] = ((zeros << 5) | (ones << 2)) as u8;
        state += 1;
    }
    table
}

/// The state-map learning rate for a slot that has visited its state
/// `n` times, as `round(65536 / (n + 1.5))` ready for a `>> 16` step.
///
/// The counter is 8 bits because a 6-bit one measured +0.4% and a 12-bit one
/// +3.6% at 1 MB: long enough to average a state's noise out, short enough
/// that a hot state keeps tracking the contexts moving through it instead of
/// freezing on a stale full-history mean.
pub const INVR: [u16; 256] = build_invr();

const fn build_invr() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut visits = 0usize;
    while visits < 256 {
        let denom = 2 * visits + 3;
        table[visits] = ((262_144 + denom) / (2 * denom)) as u16;
        visits += 1;
    }
    table
}

/// The state-map entry a slot gets on first use: the state's own frequency
/// in the high 16 bits, an empty visit counter in the low 8.
///
/// A cold context therefore starts at the count-weighted odds of its bit
/// history rather than at even odds, which is the point of keeping a history
/// state at all.
pub const SM0: [u32; 256] = build_sm0();

const fn build_sm0() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut state = 0usize;
    while state < 256 {
        let n0 = ((state >> 5) & 7) as u32;
        let n1 = ((state >> 2) & 7) as u32;
        let num = 65_536 * (2 * n1 + 1);
        let denom = 2 * (n0 + n1 + 1);
        let p = clamp16(((2 * num + denom) / (2 * denom)) as i32) as u32;
        table[state] = p << 8;
        state += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::{INVR, LOGIT_MAX, LOGIT_MIN, NX, SM0, clamp12, clamp16, clamplog, squash, stretch};

    #[test]
    fn clamps_hold_the_three_domains() {
        assert_eq!((clamp12(0), clamp12(4096)), (1, 4095));
        assert_eq!((clamp16(0), clamp16(65535)), (1, 65534));
        assert_eq!((clamplog(i32::MIN), clamplog(i32::MAX)), (-2047, 2047));
    }

    #[test]
    fn squash_saturates_and_stays_monotonic() {
        assert_eq!(squash(LOGIT_MIN), 1);
        assert_eq!(squash(0), 2048);
        assert_eq!(squash(LOGIT_MAX), 4095);
        assert_eq!(squash(-128), 1102);
        let mut last = 0;
        for logit in LOGIT_MIN..=LOGIT_MAX {
            let p = squash(logit);
            assert!(p >= last && (1..=4095).contains(&p), "{logit}: {p}");
            last = p;
        }
    }

    #[test]
    fn stretch_inverts_squash_where_the_domain_allows() {
        // The 12-bit probability has a little end of its own: past about 384
        // the two tables stop being inverses, and only monotonicity is left.
        for logit in -384..=384 {
            assert_eq!(stretch(squash(logit)), logit, "not inverse at {logit}");
        }
        let mut last = i32::MIN;
        for p in 1..4096 {
            let logit = stretch(p);
            assert!(logit >= last, "{p}: {logit} < {last}");
            last = logit;
        }
        assert_eq!((stretch(1), stretch(2048), stretch(4095)), (-1065, 0, 1065));
    }

    #[test]
    fn nx_counts_the_observed_bit() {
        assert_eq!(NX[0], 32, "a zero on an empty state is (1, 0, last 0)");
        assert_eq!(NX[1], 5, "a one on an empty state is (0, 1, last 1)");
        let saturated = (7 << 5) | (7 << 2) | 1;
        assert_eq!(NX[(saturated << 1) | 1], (6 << 5) | (7 << 2) | 1);
        for (index, &next) in NX.iter().enumerate() {
            assert_eq!(next & 1, (index & 1) as u8, "lost the observed bit");
        }
    }

    #[test]
    fn nx_decay_needs_both_counts_to_be_confident() {
        let confident = (4 << 5) | (4 << 2) | 1;
        assert_eq!(NX[(confident << 1) | 1], (3 << 5) | (5 << 2) | 1);
        let confident_zeros = (4 << 5) | (4 << 2);
        assert_eq!(NX[confident_zeros << 1], (5 << 5) | (3 << 2));
        let unsure = (3 << 5) | (4 << 2) | 1;
        assert_eq!(NX[(unsure << 1) | 1], (3 << 5) | (5 << 2) | 1);
    }

    #[test]
    fn sm0_seeds_from_the_history_it_stands_for() {
        assert_eq!(SM0[0] >> 8, 32768, "no observations are even odds");
        assert_eq!(SM0[0] & 255, 0, "the visit counter starts empty");
        assert_eq!(SM0[7 << 5] >> 8, 4096, "seven zeros, no ones");
        assert_eq!(SM0[7 << 2] >> 8, 61440, "no zeros, seven ones");
        for &entry in &SM0 {
            assert_eq!(entry & 255, 0);
            assert_eq!(clamp16((entry >> 8) as i32) as u32, entry >> 8);
        }
    }

    #[test]
    fn invr_is_the_visit_rate_scaled_to_a_shift() {
        assert_eq!(INVR[0], 43691, "1/1.5");
        assert_eq!(INVR[1], 26214, "1/2.5");
        assert_eq!(INVR[255], 256, "1/256.5 rounds to the smallest step");
        let mut last = u16::MAX;
        for rate in INVR {
            assert!(rate <= last);
            last = rate;
        }
    }
}
