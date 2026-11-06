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

//! Stage 2: the context-mixing model.
//!
//! One model instance serves both directions, which is the point of the
//! design: the encoder hands the coder a probability computed before it knows
//! the bit, and the decoder computes the same probability for the bit it has
//! not read yet. Any drift between the two is a corrupt stream, so there is
//! exactly one [`Model::predict`], one [`Model::update`] and one
//! [`Model::byte_update`].
//!
//! Eight contexts feed the mixer: hashed orders 1 to 6, a word context (the
//! alphanumeric run, so an identifier predicts as a unit) and a sparse
//! context that skips a byte - markup pairs and the fixed-width lanes of a
//! plane repeat at a stride no contiguous order can see. Each context slot
//! holds a bit-history state rather than a probability, and the probability
//! comes from a 256-entry state map per context, so every slot in the same
//! state shares one estimate: a sparse order-5 slot no longer has to infer a
//! frequency from its own two observations.
//!
//! On top of that sit a match model (the longest recent copy of the context,
//! predicting the byte after it), a two-layer logistic mixer - three
//! first-layer mixers selecting rows by bit position, by the previous byte's
//! class and by the previous byte itself, combined by a second layer - and one
//! additive-logit SSE stage.
//!
//! # Memory
//!
//! `mem_bits` sizes the context and match tables, `sse_bits` the SSE table,
//! and [`Model::memory_bytes`] reports what the result asks for. A document is
//! only worth compressing when it is larger than the model it needs, so a
//! caller checks that number before coding.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::{WercError, WercResult};
use crate::tables::{INVR, LOGIT_MAX, LOGIT_MIN, NX, SM0, clamp12, clamp16, squash, stretch};

/// Contexts mixed: six orders, one word, one sparse skip.
pub const NCTX: usize = 8;

/// Mixer inputs: the contexts plus the match model.
const NM: usize = NCTX + 1;

/// Weights per mixer row: one per input and a bias on the constant input.
const NW: usize = NM + 1;

/// Row selectors per bit position: the match model off, then on.
const NSEL: usize = 2;

/// First-layer mixers, one per row-selector width.
const NMX: usize = 3;

/// Weights per second-layer row: one per first-layer output plus a bias.
const NW2: usize = NMX + 1;

/// Rows of each first layer: ones that always learn, per byte class, per byte.
const NWR: [usize; NMX] = [8 * NSEL, 8 * NSEL * 16, 8 * NSEL * 256];

/// Rows of the second layer, selected by bit position and match state.
const M2ROWS: usize = 8 * NSEL;

/// Entries in the match model's probability table.
const MT_LEN: usize = 4096;

/// Even odds for a match-model entry that has never been trained.
const MT_INIT: u16 = 32_768;

/// Golden-ratio multiplier used to fold a context and a partial byte together.
const PHI: u32 = 0x9E37_79B9;

/// The constant input every mixer row carries a bias weight against.
const BIAS: i32 = 256;

/// Width of the longest match the model trusts.
const MATCH_MAX: usize = 16;

/// Weight a row starts each input at: the inputs pass through at unity.
const W_IN: i64 = 0x10000 / NM as i64;

/// Weight the second layer starts each first-layer output at.
const W_OUT: i64 = 0x10000 / NMX as i64;

/// Clamp a logit held as `i64` into the domain [`squash`] represents.
///
/// A dot product over ten weights is nowhere near the bound in practice; the
/// clamp is what stops a runaway weight from handing the coder a probability
/// it cannot code.
#[inline]
fn clamplog64(x: i64) -> i32 {
    if x < i64::from(LOGIT_MIN) {
        LOGIT_MIN
    } else if x > i64::from(LOGIT_MAX) {
        LOGIT_MAX
    } else {
        x as i32
    }
}

/// Squash a logit held as `i64`.
#[inline]
fn squash64(x: i64) -> i32 {
    squash(clamplog64(x))
}

/// The context-mixing model that predicts every bit of a stream.
///
/// The tables are allocated once by [`Model::new`] and never resized, and
/// every index is masked by a constructor parameter, so a coding pass neither
/// reallocates nor reads past a table.
pub struct Model {
    /// Log2 of the context and match table sizes.
    mem_bits: u32,
    /// Log2 of the SSE table size.
    sse_bits: u32,
    /// Learning-rate shift for the mixer weights.
    lr_shift: u32,
    /// Index mask for the context and match tables.
    mask: usize,
    /// Byte history: the match model's search space.
    buf: Vec<u8>,
    /// Context hashes, the most recent byte folded into `h[0]`.
    h: [u32; NCTX],
    /// Alphanumeric run hash.
    word: u32,
    /// Sparse context with one byte skipped.
    sp: u32,
    /// Bit-history state per context slot.
    tc: [Vec<u8>; NCTX],
    /// State probability map per context: `probability << 8 | visits`.
    sm: [Vec<u32>; NCTX],
    /// First-layer rows: `NWR[j]` rows of `NW` weights.
    m1: [Vec<i64>; NMX],
    /// Second-layer rows: `M2ROWS` of `NW2` weights.
    m2: Vec<i64>,
    /// Additive logit correction per order-2 context and mixed probability.
    sse: Vec<i32>,
    /// Match pointers, keyed by a five-byte hash.
    mm: Vec<u32>,
    /// Match probabilities, indexed by the match's own context.
    mt: Vec<u16>,
    /// Row each first layer read for the last prediction.
    rows: [usize; NMX],
    /// Row the second layer read for the last prediction.
    row2: usize,
    /// Mixed logits handed to the second layer.
    s2: [i32; NMX],
    /// Mixer inputs: one per context, the match model, then the bias.
    st: [i32; NW],
    /// State read from each context, written back by `update`.
    sn: [u8; NCTX],
    /// Slot read from each context, written back by `update`.
    idx: [usize; NCTX],
    /// SSE slot touched by the last prediction.
    sidx: usize,
    /// Probability the last prediction handed the coder.
    pr: i32,
    /// Read position of the current match in `buf`.
    mptr: usize,
    /// Bytes matched so far; zero means no match.
    mlen: usize,
    /// The match stopped agreeing with the bits coded so far.
    miss: bool,
}

impl Model {
    /// Build a model with `mem_bits` context rows and `sse_bits` SSE slots.
    ///
    /// # Errors
    ///
    /// [`WercError::Unsupported`] if a parameter cannot build a table:
    /// `mem_bits` outside `8..=26`, `sse_bits` outside `4..=24`, or a
    /// `lr_shift` below 9, which would shift the bias update past its own
    /// width. `lr_shift` is the one a caller is most likely to tune, so the
    /// range check is on it rather than on the memory parameters alone.
    pub fn new(mem_bits: u32, sse_bits: u32, lr_shift: u32) -> WercResult<Self> {
        if !(8..=26).contains(&mem_bits) {
            return Err(WercError::Unsupported { field: "mem_bits" });
        }
        if !(4..=24).contains(&sse_bits) {
            return Err(WercError::Unsupported { field: "sse_bits" });
        }
        if !(9..=24).contains(&lr_shift) {
            return Err(WercError::Unsupported { field: "lr_shift" });
        }
        let size = 1usize << mem_bits;
        let tc = [const { Vec::<u8>::new() }; NCTX].map(|_| vec![0u8; size]);
        let sm = [const { Vec::<u32>::new() }; NCTX].map(|_| SM0.to_vec());
        let m1 = NWR.map(|rows| {
            let mut weights = vec![0i64; rows * NW];
            // A bias of zero keeps an untrained mixer passing its inputs
            // through instead of inventing a prior of its own.
            for row in weights.chunks_mut(NW) {
                row[..NM].fill(W_IN);
            }
            weights
        });
        let mut m2 = vec![0i64; M2ROWS * NW2];
        for row in m2.chunks_mut(NW2) {
            row[..NMX].fill(W_OUT);
        }
        let mut st = [0i32; NW];
        st[NM] = BIAS;
        Ok(Self {
            mem_bits,
            sse_bits,
            lr_shift,
            mask: size - 1,
            buf: Vec::new(),
            h: [0; NCTX],
            word: 0,
            sp: 0,
            tc,
            sm,
            m1,
            m2,
            sse: vec![0i32; 1usize << sse_bits],
            mm: vec![0u32; size],
            mt: vec![MT_INIT; MT_LEN],
            rows: [0; NMX],
            row2: 0,
            s2: [0; NMX],
            st,
            sn: [0; NCTX],
            idx: [0; NCTX],
            sidx: 0,
            pr: 2048,
            mptr: 0,
            mlen: 0,
            miss: false,
        })
    }

    /// Bytes the model's tables hold.
    ///
    /// A caller deciding whether a document is worth coding compares this
    /// against its length: the model only pays for itself when the document
    /// is larger than the tables it needs.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let size = 1usize << self.mem_bits;
        let contexts = NCTX * size + NCTX * SM0.len() * 4;
        let mixer = NWR
            .iter()
            .map(|&rows| rows * NW * 8)
            .sum::<usize>()
            + M2ROWS * NW2 * 8;
        contexts + size * 4 + MT_LEN * 2 + mixer + (1usize << self.sse_bits) * 4
    }

    /// Fold a finished byte into the contexts, the word run and the match.
    ///
    /// Call it once the coder has committed a whole byte, with the byte both
    /// sides now agree on: a priming run feeds it the dictionary, an encoded
    /// run feeds it the byte that came out of the decoder.
    pub fn byte_update(&mut self, byte: u8) {
        let c = u32::from(byte);
        self.buf.push(byte);
        let h = &mut self.h;
        h[5] = h[4].wrapping_mul(PHI) ^ c.wrapping_add(0x2A6B_2E3D);
        h[4] = h[3].wrapping_mul(PHI) ^ c.wrapping_add(0x0076_B149);
        h[3] = (h[2] << 8) | c;
        h[2] = (h[1] << 8) | c;
        h[1] = ((h[0] << 8) | c) & 0xFFFF;
        h[0] = c;
        self.word = if is_word_byte(byte) {
            self.word
                .wrapping_shl(5)
                .wrapping_add(self.word)
                .wrapping_add(c)
        } else {
            0
        };
        let len = self.buf.len();
        // A context with a byte skipped over: pairs of planes and markup
        // repeat at a stride a contiguous order cannot see.
        self.sp = if len >= 4 {
            let b = &self.buf;
            (u32::from(b[len - 1]) << 16) | (u32::from(b[len - 3]) << 8) | u32::from(b[len - 4])
        } else {
            0
        };
        self.extend_match(c);
        self.miss = false;
    }

    /// Extend the running match, or look for a new one behind the last byte.
    fn extend_match(&mut self, c: u32) {
        let len = self.buf.len();
        if self.mlen != 0 {
            if self.mptr < len && self.buf[self.mptr] == byte_of(c) {
                self.mptr += 1;
                self.mlen += 1;
            } else {
                self.mlen = 0;
            }
        }
        if self.mlen >= 6 {
            return;
        }
        let key = (self.h[0] ^ (self.h[1] << 1) ^ (self.h[2] << 2) ^ (self.h[3] << 3)) as usize & self.mask;
        // The pointer-table miss and, once the match is armed, the follow-up
        // read of `buf[mptr]` in `match_input` are the only random loads on
        // the byte boundary; both are hinted the moment their address is
        // known.
        // SAFETY: `key` is masked below the table's length.
        unsafe { crate::simd::touch(self.mm.as_ptr().cast::<u8>(), key * 4) };
        // A stream body is a `u32` long by the container's own limit, so the
        // history position always fits the pointer table.
        let position = self.buf.len() as u32;
        let cand = self.mm[key] as usize;
        self.mm[key] = position;
        // The key is a five-byte hash, so confirm it against the bytes behind
        // the pointer: a collision would otherwise spend the next byte on a
        // prediction that has nothing to do with the context.
        if cand != 0 && cand != position as usize && cand >= 2 && self.buf[cand - 1] == byte_of(c) {
            let previous = byte_of((self.h[1] >> 8) & 0xFF);
            if self.buf[cand - 2] == previous {
                self.mptr = cand;
                self.mlen = 1;
            }
        }
        if self.mlen != 0 && self.mptr < self.buf.len() {
            // SAFETY: the bounds test above is the function's own; the hint
            // reads the same byte `match_input` will.
            unsafe { crate::simd::touch(self.buf.as_ptr(), self.mptr) };
        }
    }

    /// The match model's input to the mixer for the bit being coded.
    fn match_input(&mut self, c0: u32, nbits: u32) -> i32 {
        if self.mlen == 0 || self.miss || self.mptr >= self.buf.len() {
            return 0;
        }
        let predicted = u32::from(self.buf[self.mptr]);
        let sofar = c0 ^ (1 << nbits);
        if nbits != 0 && (predicted >> (8 - nbits)) != sofar {
            self.miss = true;
            return 0;
        }
        let bit = (predicted >> (7 - nbits)) & 1;
        let bucket = if self.mlen < MATCH_MAX {
            self.mlen
        } else {
            MATCH_MAX
        };
        let i = (((bucket << 4) | ((nbits as usize) << 1) | bit as usize) << 4 | (sofar & 15) as usize)
            & (MT_LEN - 1);
        let seen = u32::from(self.mt[i]);
        let step = (((bit << 16) as i32) - seen as i32) >> 6;
        let updated = clamp16(seen as i32 + step) as u16;
        self.mt[i] = updated;
        // The mixer sees the probability the match had *before* this bit, so
        // the encoder and the decoder both credit it for the same prediction.
        stretch(clamp12((seen >> 4) as i32))
    }

    /// The 12-bit probability that the next bit is a one.
    ///
    /// `c0` is the partial byte carrying its leading `1` marker, so `1..=255`
    /// for `nbits` in `0..8`. The decoder knows both before it reads the bit,
    /// which is what lets one method serve two directions. A `nbits` outside
    /// `0..8` is masked rather than rejected: the byte loop is the only
    /// caller, and a panic here would abort a coding pass.
    pub fn predict(&mut self, c0: u32, nbits: u32) -> i32 {
        let nbits = nbits & 7;
        let pc0 = c0.wrapping_mul(PHI);
        let sel = usize::from(self.mlen != 0 && !self.miss);
        let row = (nbits as usize) * NSEL + sel;
        self.row2 = row;
        let prev = (self.h[0] & 0xFF) as usize;
        self.rows = [row, (row << 4) | (prev >> 4), (row << 8) | prev];
        // The slot reads and the mixer rows below are the misses this loop
        // pays for; their addresses are independent, so every hint is issued
        // before the first load that needs it and the misses overlap.
        let mut slots = [0usize; NCTX];
        for (slot, h) in slots[..6].iter_mut().zip(self.h) {
            *slot = (h ^ pc0).wrapping_mul(PHI) as usize & self.mask;
        }
        slots[6] = (self.word ^ pc0) as usize & self.mask;
        slots[7] = (self.sp ^ pc0) as usize & self.mask;
        for (table, i) in self.tc.iter().zip(slots) {
            // SAFETY: every slot index is masked below `self.mask`, and each
            // `tc` table is `mask + 1` bytes long.
            unsafe { crate::simd::touch(table.as_ptr(), i) };
        }
        for j in 0..NMX {
            let byte_row = self.rows[j] * (NW * 8);
            let base = self.m1[j].as_ptr().cast::<u8>();
            // SAFETY: a row is `NW` i64s = 88 bytes at `byte_row`, which is
            // `rows * NW * 8` bytes of table; `byte_row + 64` stays inside
            // the row, hence inside the allocation.
            unsafe {
                crate::simd::touch(base, byte_row);
                crate::simd::touch(base, byte_row + 64);
            }
        }
        for (k, &i) in slots.iter().enumerate() {
            self.read_slot(k, i);
        }
        self.st[NCTX] = self.match_input(c0, nbits);
        let base = row * NW2;
        let mut dot2 = 0i64;
        for j in 0..NMX {
            let row0 = self.rows[j] * NW;
            let mut dot = 0i64;
            for i in 0..NW {
                dot = dot.wrapping_add(self.m1[j][row0 + i].wrapping_mul(i64::from(self.st[i])));
            }
            let mixed = stretch(squash64(dot >> 16));
            self.s2[j] = mixed;
            dot2 = dot2.wrapping_add(self.m2[base + j].wrapping_mul(i64::from(mixed)));
        }
        let logit = clamplog64((dot2 + (self.m2[base + NMX] << 8)) >> 16);
        let mixed = squash(logit);
        // One SSE stage keyed by the order-2 context and the mixed
        // probability. It starts at zero, so an untrained table leaves the
        // mixer alone instead of pulling every prediction halfway to even.
        self.sidx = ((self.idx[1] >> 4) ^ ((mixed >> 6) as usize)) & (self.sse.len() - 1);
        let correction = i64::from(self.sse[self.sidx] >> 6);
        self.pr = squash64(i64::from(logit) + correction);
        self.pr
    }

    /// Read one context slot, keeping the state and index the update needs.
    #[inline]
    fn read_slot(&mut self, k: usize, i: usize) {
        self.idx[k] = i;
        let state = self.tc[k][i];
        self.sn[k] = state;
        self.st[k] = stretch((self.sm[k][usize::from(state)] >> 12) as i32);
    }

    /// Adapt the state maps, the mixer and the SSE stage to the coded `bit`.
    ///
    /// Call it with the bit the coder just committed, before the next
    /// [`Model::predict`].
    pub fn update(&mut self, bit: u32) {
        let bit = bit & 1;
        let err = ((bit as i32) << 12) - self.pr;
        let sh = self.lr_shift;
        let base = self.row2 * NW2;
        for j in 0..NMX {
            let step = (i64::from(err).wrapping_mul(i64::from(self.s2[j]))) >> sh;
            self.m2[base + j] = self.m2[base + j].wrapping_add(step);
        }
        self.m2[base + NMX] = self.m2[base + NMX].wrapping_add(i64::from(err) >> (sh - 8));
        // The stretch that fed the second layer is the inverse of the squash
        // that produced it, so those two derivatives cancel and each
        // first-layer weight sees the second layer's error scaled by that
        // mixer's own output weight. Boosting that scale measured worse: the
        // narrow row is what keeps the wide selectors readable while they are
        // still learning.
        for j in 0..NMX {
            let e1 = (i64::from(err).wrapping_mul(self.m2[base + j])) >> 16;
            if e1 == 0 {
                continue;
            }
            let row = self.rows[j] * NW;
            for i in 0..NW {
                let step = e1.wrapping_mul(i64::from(self.st[i])) >> sh;
                self.m1[j][row + i] = self.m1[j][row + i].wrapping_add(step);
            }
        }
        let target = i64::from(bit) << 16;
        for k in 0..NCTX {
            let state = usize::from(self.sn[k]);
            let entry = self.sm[k][state];
            let p = (entry >> 8) as i32;
            let visits = (entry & 255) as usize;
            let step = ((target - i64::from(p)) * i64::from(INVR[visits])) >> 16;
            let p = clamp16(p + step as i32) as u32;
            self.sm[k][state] = (p << 8) | (visits + usize::from(visits < 255)) as u32;
            self.tc[k][self.idx[k]] = NX[(state << 1) | bit as usize];
        }
        self.sse[self.sidx] = self.sse[self.sidx].wrapping_add(err >> 5);
    }

    /// Forget the byte history and the match, keeping everything learned.
    ///
    /// A primed dictionary shapes the statistics but must not become the match
    /// model's history: the decoder cannot see it, so a match found in the
    /// dictionary would predict bytes no one can confirm.
    pub fn reset_history(&mut self) {
        self.buf.clear();
        self.mm.iter_mut().for_each(|slot| *slot = 0);
        self.mlen = 0;
        self.mptr = 0;
        self.miss = false;
    }
}

/// Whether `byte` continues an identifier run.
#[inline]
const fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_digit() || byte.is_ascii_alphabetic() || byte == b'_'
}

/// The low byte of a context word, as the match model compares it.
#[inline]
const fn byte_of(value: u32) -> u8 {
    value as u8
}

#[cfg(test)]
mod tests {
    use super::{Model, NCTX};
    use crate::tables::SM0;

    /// A source fragment: enough structure for the word, order and match
    /// contexts to disagree with each other.
    const CORPUS: &[u8] = b"fn main() {\n    let x = 1 + 2;\n}\n";

    /// Code `data` through the model and return the probability handed to the
    /// coder for every bit, which is the whole observable behaviour of a
    /// model: the coder is given here, so any divergence shows up in these.
    fn predictions(model: &mut Model, data: &[u8]) -> Vec<i32> {
        let mut out = Vec::with_capacity(data.len() * 8);
        for &byte in data {
            let mut c0 = 1u32;
            for nbits in 0..8u32 {
                out.push(model.predict(c0, nbits));
                let bit = (u32::from(byte) >> (7 - nbits)) & 1;
                model.update(bit);
                c0 = (c0 << 1) | bit;
            }
            model.byte_update(byte);
        }
        out
    }

    #[test]
    fn predictions_match_the_python_reference() {
        let mut model = Model::new(12, 12, 11).expect("test parameters are in range");
        let coded = predictions(&mut model, CORPUS);
        assert_eq!(
            &coded[..16],
            [
                2048, 786, 2048, 2557, 2048, 1715, 2048, 2279, 1904, 2016, 2318, 1896, 1699, 2168, 2287, 2381
            ],
            "the model diverged from the Python reference"
        );
        assert_eq!(
            &coded[coded.len() - 8..],
            [728, 1731, 1944, 1645, 1645, 1443, 1584, 1479]
        );
        assert_eq!(coded.iter().sum::<i32>(), 426_653);
    }

    #[test]
    fn priming_learns_without_leaking_the_dictionary_history() {
        let mut model = Model::new(12, 12, 11).expect("test parameters are in range");
        predictions(&mut model, &CORPUS[..8]);
        model.reset_history();
        assert!(model.buf.is_empty());
        assert_eq!(model.mlen, 0);
        let coded = predictions(&mut model, CORPUS);
        assert_eq!(
            coded.iter().sum::<i32>(),
            424_184,
            "the statistics survived the reset"
        );
    }

    #[test]
    fn every_prediction_is_codable() {
        let mut model = Model::new(12, 12, 11).expect("test parameters are in range");
        for p in predictions(&mut model, CORPUS) {
            assert!((1..=4095).contains(&p), "the coder cannot split at {p}");
        }
    }

    #[test]
    fn a_repeated_run_arms_the_match_model() {
        // What the match model is for: once the context recurs, the second run
        // of the same text should cost fewer bits than the first, which only
        // happens if the match feeds the mixer something the cold start lacks.
        let repeated = b"let value = compute(a, b);\nlet value = compute(a, b);\n";
        let half = repeated.len() / 2;
        let mut model = Model::new(14, 12, 11).expect("test parameters are in range");
        let mut cost = [0f64; 2];
        for (i, &byte) in repeated.iter().enumerate() {
            let side = usize::from(i >= half);
            let mut c0 = 1u32;
            for nbits in 0..8u32 {
                let p = f64::from(model.predict(c0, nbits)) / 4096.0;
                let bit = (u32::from(byte) >> (7 - nbits)) & 1;
                cost[side] -= (if bit == 1 { p } else { 1.0 - p }).log2();
                model.update(bit);
                c0 = (c0 << 1) | bit;
            }
            model.byte_update(byte);
        }
        assert!(
            cost[1] < cost[0],
            "cold start cost {:.1} bits, the repeat {:.1}",
            cost[0],
            cost[1]
        );
    }

    #[test]
    fn out_of_range_parameters_are_rejected() {
        assert_eq!(
            Model::new(7, 12, 11).err().map(|e| e.label()),
            Some("unsupported stream parameter")
        );
        for (mem, sse, lr) in [(27, 12, 11), (12, 3, 11), (12, 25, 11), (12, 12, 8)] {
            assert!(
                Model::new(mem, sse, lr).is_err(),
                "{mem} {sse} {lr} should be refused"
            );
        }
    }

    #[test]
    fn memory_bytes_counts_every_table() {
        let model = Model::new(12, 12, 11).expect("test parameters are in range");
        let tables = model.tc.iter().map(|t| t.len()).sum::<usize>()
            + model
                .sm
                .iter()
                .map(|t| t.len() * 4)
                .sum::<usize>()
            + model.mm.len() * 4
            + model.mt.len() * 2
            + model
                .m1
                .iter()
                .map(|t| t.len() * 8)
                .sum::<usize>()
            + model.m2.len() * 8
            + model.sse.len() * 4;
        assert_eq!(model.memory_bytes(), tables);
        assert_eq!(model.tc.len(), NCTX);
        assert_eq!(model.sm[0].len(), SM0.len());
    }
}
