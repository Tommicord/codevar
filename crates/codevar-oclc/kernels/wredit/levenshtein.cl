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
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// OR CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing
// permissions and limitations under the License.

// Change-distance kernel: capped Levenshtein distance between two
// code-point sequences.
//
// The Wredit engine compares two versions of a change (for example
// the same edit seen by two collaborators, or an edit that mostly
// moves whitespace) and asks one question: are these changes close
// enough to blend instead of treating them as conflicting? That
// question is an edit distance with a *cap* — callers only ever need
// to know the distance up to `max_dist`, beyond which every answer
// means "too different".
//
// The kernel computes the Wagner-Fischer dynamic program with a
// banded optimization: only cells with `|row - col| <= max_dist` are
// evaluated, because any path leaving that band already costs more
// than the cap. Cells beyond the band behave as if they held
// `max_dist + 1`, the *saturation* value, which doubles as the answer
// itself for "further apart than the cap". This keeps the work at
// `O(a_len * (2 * max_dist + 1))` instead of `O(a_len * b_len)` and
// lets the row minimum act as a monotone early exit: the smallest
// cell in a row never decreases across rows, so once it passes the
// cap no later row can bring the final distance back within it.
//
// One work item owns the whole dynamic program (a single sequence
// pair), mirroring how a batched GPU implementation would assign one
// thread per pair. The two DP rows live in `work`, a scratch buffer
// the host sizes at `2 * (b_len + 1)` integers; row `r` alternates
// between the two halves, so `prev` and `cur` are always one row
// apart and no synchronization is needed. The host launches a global
// size of one and reads the single result from `out[0]`.
//
// Launch with a global size of exactly one. The host guarantees
// `a_len`, `b_len` and `max_dist` are far below `i32::MAX / 2`, so
// every index and `row + max_dist` sum stays well inside 32 bits.

// Distance between `a[0..a_len]` and `b[0..b_len]`, saturated at
// `max_dist + 1`.
//
// Returns `min(levenshtein(a, b), max_dist + 1)`: the exact distance
// when it is at most `max_dist`, and `max_dist + 1` otherwise. The
// host allocates `work` with room for `2 * (b_len + 1)` integers and
// keeps both inputs alive until the kernel event has been waited on.
#[kernel]
fn levenshtein(
    a: *const int,
    a_len: int,
    b: *const int,
    b_len: int,
    max_dist: int,
    work: *mut int,
    out: *mut int,
) -> void {
    let sat = max_dist + 1;
    let row_len = b_len + 1;

    // Row 0: turning the empty prefix of `a` into `b[0..j]` costs `j`
    // insertions, saturated at the cap. Cells past `top` stay
    // untouched; the band guards below never read them.
    let mut top = b_len;
    if max_dist < b_len {
        top = max_dist;
    }
    let mut j = 0;
    while j <= top {
        work[j] = j;
        j = j + 1;
    }

    let mut r = 1;
    let mut far = 0;
    while r <= a_len && far == 0 {
        // Row `r` uses the second half when odd, the first when even.
        let mut cur = 0;
        if r - (r / 2) * 2 == 1 {
            cur = row_len;
        }
        let prev = row_len - cur;

        // Band of the current row: cells outside `[lo, hi]` cannot
        // lead to a distance within the cap.
        let mut lo = 0;
        if r > max_dist {
            lo = r - max_dist;
        }
        let hi_cap = r + max_dist;
        let mut hi = hi_cap;
        if b_len < hi_cap {
            hi = b_len;
        }

        // Band of the previous row: the only `prev` cells the three
        // recurrence terms are allowed to read. A read outside this
        // band would see a stale value, so it is replaced by `sat`.
        let mut p_lo = 0;
        if r - 1 > max_dist {
            p_lo = r - 1 - max_dist;
        }
        let p_hi_cap = r - 1 + max_dist;
        let mut p_hi = p_hi_cap;
        if b_len < p_hi_cap {
            p_hi = b_len;
        }

        // Smallest cell written in this row; never increases between
        // rows, which is what makes the early exit below sound.
        let mut row_min = sat;
        if lo == 0 {
            // D(r, 0) = r insertions, saturated.
            let mut first = sat;
            if r < sat {
                first = r;
            }
            work[cur] = first;
            if first < row_min {
                row_min = first;
            }
        }

        let mut col = lo;
        if col < 1 {
            col = 1;
        }
        while col <= hi {
            let cost = if a[r - 1] != b[col - 1] { 1 } else { 0 };
            // substitution: D(r - 1, col - 1) + (a[r-1] != b[col-1])
            let mut sub = sat;
            if col - 1 >= p_lo {
                sub = min(sat, work[prev + col - 1] + cost);
            }
            // deletion: D(r - 1, col) + 1
            let mut del = sat;
            if col <= p_hi {
                del = min(sat, work[prev + col] + 1);
            }
            // insertion: D(r, col - 1) + 1
            let mut ins = sat;
            if col - 1 >= lo {
                ins = min(sat, work[cur + col - 1] + 1);
            }
            let value = min(sub, min(del, ins));
            work[cur + col] = value;
            if value < row_min {
                row_min = value;
            }
            col = col + 1;
        }

        if row_min > max_dist {
            far = 1;
        }
        r = r + 1;
    }

    // The bottom-right cell only holds the answer when `b_len` sits
    // inside the last row's band; a diagonal gap wider than the cap
    // already answers `sat`.
    let mut diff = a_len - b_len;
    if b_len > a_len {
        diff = b_len - a_len;
    }
    if far == 1 || diff > max_dist {
        out[0] = sat;
    } else {
        let mut last = 0;
        if a_len - (a_len / 2) * 2 == 1 {
            last = row_len;
        }
        out[0] = work[last + b_len];
    }
}
