// Copyright 2026 Codevar Project
// Licensed under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the
// License at
//
//   https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in
// writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

//! The logistic domain of the codec: where a probability and a logit meet.
//!
//! Three widths are in play. The coder codes 12-bit probabilities
//! (`1..=4095`), the mixer accumulates logits as `i32`, and the state map
//! keeps 16-bit probabilities. [`squash`] and [`stretch`] move between the
//! first two, [`clamp12`] and [`clamplog`] keep each inside the domain the
//! other one's table was built for.
//!
//! Both tables are generated data - the `SQ` and `ST` arrays of
//! `tables_data` - because a build-time computation would make a stream
//! depend on the rounding of the host libm. They are public because they
//! *are* the format: a stream is only decodable by a model that reads the
//! same squash curve.

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

#[cfg(test)]
mod tests {
    use super::{LOGIT_MAX, LOGIT_MIN, clamp12, clamplog, squash, stretch};

    #[test]
    fn clamps_hold_the_two_domains() {
        assert_eq!((clamp12(0), clamp12(4096)), (1, 4095));
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
}
