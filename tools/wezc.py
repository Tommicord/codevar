#!/usr/bin/env python3
# Copyright 2026 Codevar Project
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# https://www.apache.org/licenses/LICENSE-2.0
#
# WEZC (Wredit Zstd Compression) - first Python prototype.
#
# The design study said the packlow framing is worth keeping only as a
# *transform* stage, and that the ratio has to come from modelling, so this
# prototype is three stages:
#
#   stage 1  codepoint transform : UTF-32 -> dense plane0 + delta-coded
#                                  escape list. The fixed-width framing
#                                  carries almost no information in planes
#                                  1-3 for real text, so it is removed
#                                  before any modelling happens.
#   stage 2  dictionary priming  : an optional corpus is run through the
#                                  model without emitting bytes, which is
#                                  what buys ratio at the 4 KiB block size
#                                  of the Wredit store.
#   stage 3  context mixing      : order 1-6 contexts + word context
#                                  + a match model, logistic mixing with
#                                  match-selected weight sets, one SSE
#                                  stage, binary arithmetic coder.
#
# Everything is integer only and the encoder and the decoder share one
# coding loop, so the two cannot drift apart.
#
# Usage:
#   wezc.py roundtrip  FILE [--mode auto|raw|utf32] [--limit N] [--dict D]
#   wezc.py compress   FILE [-o OUT]  [...]
#   wezc.py decompress FILE [-o OUT]  [...]
#   wezc.py bench      FILE [...]      # encode + decode + verify, bpc and rate
"""WEZC prototype: codepoint transform + dictionary priming + context mixing."""

import argparse
import math
import sys
import time
from array import array

MAGIC = b"WEZC\x01"

MODE_RAW = 1
MODE_UTF32 = 2

#: bytes: magic, mode, mem, sse, lr, flags, original length, units, body length
HDR_SIZE = 22

MASK32 = 0xFFFFFFFF
PHI = 0x9E3779B9

# --- probability tables -----------------------------------------------------
# squash: logit index [-2047..2047] -> 12-bit probability, stretch: inverse.
SQ = array("h", bytes(4096 * 2))
ST = array("h", bytes(4096 * 2))
for _i in range(-2047, 2048):
    SQ[_i + 2047] = max(1, min(4095, int(round(4096.0 / (1.0 + math.exp(-_i / 128.0))))))
for _p in range(1, 4096):
    ST[_p] = max(-2047, min(2047, int(round(128.0 * math.log(_p / (4096.0 - _p))))))


def clamp12(p):
    """Keep a 12-bit probability inside the domain the tables were built for."""
    if p < 1:
        return 1
    if p > 4095:
        return 4095
    return p


def clamplog(x):
    """Keep a logit inside the domain the squash table was built for."""
    if x < -2047:
        return -2047
    if x > 2047:
        return 2047
    return x


def clamp16(p):
    if p < 1:
        return 1
    if p > 65534:
        return 65534
    return p


# --- stage 1: UTF-32 plane split -------------------------------------------


def utf32_split(raw):
    """Split UTF-32LE bytes into plane0 plus a delta-coded escape list.

    Layout of the returned body: codepoint count, escape count, the escape
    positions as varint deltas, the escape remainders as 3-byte little
    endian words, then plane0.
    """
    if len(raw) % 4:
        raise ValueError("utf32 mode needs a multiple of 4 bytes")
    n = len(raw) >> 2
    plane0 = raw[0::4]
    p1, p2, p3 = raw[1::4], raw[2::4], raw[3::4]
    esc = [k for k in range(n) if p1[k] or p2[k] or p3[k]]
    out = bytearray()
    out += n.to_bytes(4, "little")
    out += len(esc).to_bytes(4, "little")
    prev = 0
    for k in esc:
        d = k - prev
        prev = k
        while d >= 0x80:
            out.append((d & 0x7F) | 0x80)
            d >>= 7
        out.append(d)
    for k in esc:
        out += (p1[k] | (p2[k] << 8) | (p3[k] << 16)).to_bytes(3, "little")
    out += plane0
    return bytes(out), n


def utf32_join(body):
    """Rebuild the UTF-32LE bytes that :func:`utf32_split` consumed."""
    n = int.from_bytes(body[:4], "little")
    cnt = int.from_bytes(body[4:8], "little")
    i = 8
    pos = []
    prev = 0
    for _ in range(cnt):
        d, shift = 0, 0
        while True:
            b = body[i]
            i += 1
            d |= (b & 0x7F) << shift
            if b & 0x80:
                shift += 7
            else:
                break
        prev += d
        pos.append(prev)
    values = [int.from_bytes(body[i + 3 * k : i + 3 * k + 3], "little") for k in range(cnt)]
    i += 3 * cnt
    plane0 = body[i : i + n]
    raw = bytearray(4 * n)
    raw[0::4] = plane0
    for k, p in enumerate(pos):
        v = values[k]
        raw[4 * p + 1] = v & 0xFF
        raw[4 * p + 2] = (v >> 8) & 0xFF
        raw[4 * p + 3] = (v >> 16) & 0xFF
    return bytes(raw)


# --- stage 3: the context-mixing model ------------------------------------

# context tables: orders 1, 2, 3, 4, 5, 6 (hashed) and the word context
NCTX = 7
# mixer inputs: the context tables plus the match model
NM = NCTX + 1
# mixer weights: one per input plus a bias
NW = NM + 1
# mixer weight sets per bit position: the match model off, then on
NSEL = 2

# A context keeps its 16-bit probability together with how many times it has
# been seen: the learning rate is 1/(count+2), so a fresh slot moves in big
# steps and a well-evidenced one barely moves. Fixed rates leave the sparse
# order-4 and order-5 slots near 0.5 forever, which is where the ratio is.
CMAX = 32
INV = array("H", [0] * (CMAX + 1))
for _n in range(CMAX + 1):
    INV[_n] = int(round(65536.0 / (_n + 2)))
CINIT = (32768 << 6)  # count 0, p = 0.5
CMASK = 63


class Wezc:
    """One model instance; the encoder and the decoder drive it identically."""

    def __init__(self, mem_bits=22, sse_bits=16, lr_shift=11):
        size = 1 << mem_bits
        self.mask = size - 1
        # every hashed context starts at p(bit=1) = 0.5
        self.t = [array("I", (CINIT,)) * size for _ in range(NCTX)]
        # Mixer weight sets, selected per bit by what is already known about
        # it: the partial byte (which bits are decided and how) and the match
        # model's confidence. One row per bit position, as before, has to
        # average a confident match copy together with a bare order-1 guess,
        # and the two want different input weights.
        self.wrows = [array("q", [0x10000 // NM] * NM + [0])
                      for _ in range(8 * NSEL)]
        self.w = self.wrows[0]
        self.lr_shift = lr_shift
        self.sse = array("i", bytes(4 << sse_bits))
        self.sse_mask = (1 << sse_bits) - 1
        self.h = [0] * NCTX
        self.word = 0
        # match model
        self.mm_bits = mem_bits
        self.mm = array("I", bytes((1 << self.mm_bits) * 4))
        self.mt = array("H", b"\x00\x80") * 4096
        self.buf = bytearray()
        self.mptr = 0
        self.mlen = 0
        self.miss = False
        # per-bit scratch, filled by predict and consumed by update
        self.idx = [0] * NCTX
        self.st = [0] * NM
        self.probs = [32768] * NCTX
        self.nc = [0] * NCTX
        self.sidx = 0
        self.pr = 2048

    def byte_update(self, c):
        """Fold a finished byte into the contexts and the match model."""
        self.buf.append(c)
        h = self.h
        h[5] = ((h[4] * PHI) ^ (c + 0x2A6B2E3D)) & MASK32  # order 6, hashed
        h[4] = ((h[3] * PHI) ^ (c + 0x76B149)) & MASK32  # order 5, hashed
        h[3] = ((h[2] << 8) | c) & MASK32  # order 4, exact
        h[2] = ((h[1] << 8) | c) & MASK32  # order 3, exact
        h[1] = ((h[0] << 8) | c) & 0xFFFF  # order 2, exact
        h[0] = c
        if 48 <= c <= 57 or 65 <= c <= 90 or 97 <= c <= 122 or c == 95:
            self.word = ((self.word << 5) + self.word + c) & MASK32
        else:
            self.word = 0
        # match model: extend while the predicted byte keeps agreeing
        if self.mlen:
            if self.mptr < len(self.buf) and self.buf[self.mptr] == c:
                self.mptr += 1
                self.mlen += 1
            else:
                self.mlen = 0
        if self.mlen < 6:
            key = (h[0] ^ (h[1] << 1) ^ (h[2] << 2) ^ (h[3] << 3)) & ((1 << self.mm_bits) - 1)
            cand = self.mm[key]
            self.mm[key] = len(self.buf)
            # the key is a 5-byte hash, so confirm it against the bytes behind
            # the pointer; a collision would otherwise spend the next byte on
            # a prediction that has nothing to do with the context.
            if cand and cand != len(self.buf) and cand >= 2:
                if self.buf[cand - 1] == c and self.buf[cand - 2] == (h[1] >> 8):
                    self.mptr = cand
                    self.mlen = 1
        self.miss = False

    def _match_input(self, c0, nbits):
        """Logit contribution of the match model for the bit being coded."""
        if not self.mlen or self.miss:
            return 0
        if self.mptr >= len(self.buf):
            return 0
        pred = self.buf[self.mptr]
        sofar = c0 ^ (1 << nbits)
        if nbits and (pred >> (8 - nbits)) != sofar:
            self.miss = True
            return 0
        bit = (pred >> (7 - nbits)) & 1
        bucket = self.mlen if self.mlen < 16 else 16
        idx = (((bucket << 4) | (nbits << 1) | bit) << 4) | (sofar & 15)
        idx &= 4095
        p = self.mt[idx]
        self.mt[idx] = clamp16(p + (((bit << 16) - p) >> 6))
        return ST[clamp12(p >> 4)]

    def predict(self, c0, nbits):
        """Return the 12-bit P(bit=1) for the next bit and stash the state."""
        m = self.mask
        h = self.h
        idx = self.idx
        st = self.st
        probs = self.probs
        t = self.t
        dot = 0
        pc0 = (c0 * PHI) & MASK32
        ml = self.mlen
        sel = 1 if (ml and not self.miss) else 0
        self.w = w = self.wrows[nbits * NSEL + sel]
        nc = self.nc
        for k in range(6):
            # multiply-xor, so that different partial bytes never land on the
            # same slot: a plain `h ^ c0` folds 256 contexts into one index.
            i = (((h[k] ^ pc0) * PHI) & MASK32) & m
            idx[k] = i
            e = t[k][i]
            p = e >> 6
            nc[k] = e & CMASK
            probs[k] = p
            s = ST[clamp12(p >> 4)]
            st[k] = s
            dot += w[k] * s
        i = (self.word ^ pc0) & m
        idx[6] = i
        e = t[6][i]
        p = e >> 6
        nc[6] = e & CMASK
        probs[6] = p
        s = ST[clamp12(p >> 4)]
        st[6] = s
        dot += w[6] * s
        mn = self._match_input(c0, nbits)
        st[NCTX] = mn
        dot += w[NCTX] * mn + w[NM] * 256
        logit = dot >> 16
        if logit < -2047:
            logit = -2047
        elif logit > 2047:
            logit = 2047
        p = SQ[logit + 2047]
        # one SSE stage: an additive logit correction keyed by the order-2
        # context and the mixed probability. It starts at zero, so an
        # untrained table leaves the mixer alone instead of pulling every
        # prediction halfway to 0.5.
        sidx = ((idx[1] >> 4) ^ (p >> 6)) & self.sse_mask
        self.sidx = sidx
        self.pr = SQ[clamplog(logit + (self.sse[sidx] >> 6)) + 2047]
        return self.pr

    def update(self, bit):
        """Adapt the context tables, the mixer and the SSE stage."""
        err = (bit << 12) - self.pr
        sh = self.lr_shift
        w = self.w
        st = self.st
        probs = self.probs
        idx = self.idx
        nc = self.nc
        target = bit << 16
        for k in range(NCTX):
            n = nc[k] + 1
            if n > CMAX:
                n = CMAX
            p = clamp16(probs[k] + (((target - probs[k]) * INV[n]) >> 16))
            self.t[k][idx[k]] = (p << 6) | n
            w[k] += (err * st[k]) >> sh
        w[NCTX] += (err * st[NCTX]) >> sh
        w[NM] += err >> (sh - 8)
        self.sse[self.sidx] += err >> 5


# --- the shared coding loop -------------------------------------------------


def _run(model, blocks, encode, payload):
    """Code (or decode) `blocks` with `model`.

    A block is `(source, count, priming)`. `source` is always available for
    a priming block - both sides own the dictionary - and for the encoded
    block; when it is `None` the bytes come out of the arithmetic decoder.
    A priming block is folded into the model without the coder touching the
    stream.
    """
    x1, x2 = 0, MASK32
    sink = bytearray()
    pos = 4
    x = int.from_bytes(payload[0:4].ljust(4, b"\x00"), "big") if not encode else 0
    for src, count, priming in blocks:
        # the bits fed to the model are known whenever the bytes are known
        from_input = encode or priming
        for i in range(count):
            byte_in = src[i] if from_input else 0
            c0, db = 1, 0
            for nbits in range(8):
                p = model.predict(c0, nbits)
                bit = (byte_in >> (7 - nbits)) & 1
                if not priming:
                    mid = x1 + (((x2 - x1) >> 12) * p)
                    if encode:
                        if bit:
                            x2 = mid
                        else:
                            x1 = mid + 1
                        while not (x1 ^ x2) & 0xFF000000:
                            sink.append((x1 >> 24) & 0xFF)
                            x1 = (x1 << 8) & MASK32
                            x2 = ((x2 << 8) | 0xFF) & MASK32
                    else:
                        bit = 1 if x <= mid else 0
                        if bit:
                            x2 = mid
                        else:
                            x1 = mid + 1
                        while not (x1 ^ x2) & 0xFF000000:
                            x1 = (x1 << 8) & MASK32
                            x2 = ((x2 << 8) | 0xFF) & MASK32
                            x = ((x << 8) | (payload[pos] if pos < len(payload) else 0)) & MASK32
                            pos += 1
                model.update(bit)
                c0 = (c0 << 1) | bit
                db = (db << 1) | bit
            if not encode and not priming:
                sink.append(db)
            model.byte_update(byte_in if from_input else db)
        if priming:
            # the dictionary primes the statistics but must not become the
            # match model's history - the decoder cannot see it either.
            model.buf.clear()
            model.mlen = 0
            model.mm = array("I", bytes((1 << model.mm_bits) * 4))
    if encode:
        for _ in range(4):
            sink.append((x1 >> 24) & 0xFF)
            x1 = (x1 << 8) & MASK32
    return bytes(sink)


def encode_stream(data, dictionary=b"", mem_bits=22, sse_bits=16, lr_shift=11):
    model = Wezc(mem_bits, sse_bits, lr_shift)
    blocks = [(dictionary, len(dictionary), True)] if dictionary else []
    blocks.append((data, len(data), False))
    return _run(model, blocks, True, None)


def decode_stream(payload, length, dictionary=b"", mem_bits=22, sse_bits=16, lr_shift=11):
    model = Wezc(mem_bits, sse_bits, lr_shift)
    blocks = [(dictionary, len(dictionary), True)] if dictionary else []
    blocks.append((None, length, False))
    return _run(model, blocks, False, payload)


# --- container --------------------------------------------------------------


def pack(data, mode, dictionary=b"", mem_bits=22, sse_bits=16, lr_shift=11):
    """Return (blob, original length, model bytes, codepoints)."""
    if mode == MODE_UTF32:
        body, units = utf32_split(data)
    else:
        body, units = data, len(data)
    coded = encode_stream(body, dictionary, mem_bits, sse_bits, lr_shift)
    hdr = bytearray(MAGIC)
    hdr.append(mode)
    hdr.append(mem_bits)
    hdr.append(sse_bits)
    hdr.append(lr_shift)
    hdr.append(0)
    hdr += len(data).to_bytes(4, "little")
    hdr += units.to_bytes(4, "little")
    hdr += len(body).to_bytes(4, "little")
    return bytes(hdr) + coded, len(data), len(body), units


def unpack(blob, dictionary=b""):
    if blob[:5] != MAGIC:
        raise ValueError("not a WEZC stream")
    mode = blob[5]
    mem_bits = blob[6]
    sse_bits = blob[7]
    lr_shift = blob[8]
    orig_len = int.from_bytes(blob[10:14], "little")
    units = int.from_bytes(blob[14:18], "little")
    body_len = int.from_bytes(blob[18:22], "little")
    if body_len > orig_len * 4 + 16:
        raise ValueError("implausible body length")
    body = decode_stream(
        blob[HDR_SIZE:], body_len, dictionary, mem_bits, sse_bits, lr_shift
    )
    data = utf32_join(body) if mode == MODE_UTF32 else body
    return data, orig_len, units


# --- command line -----------------------------------------------------------


def _selftest(mem=16, sse=12):
    """Round-trip synthetic inputs; returns the number of failures."""
    cases = [
        ("empty", b"", MODE_RAW),
        ("one byte", b"\x41", MODE_RAW),
        ("all zero", bytes(9000), MODE_RAW),
        ("repeated text", (b"the quick brown fox\n" * 400), MODE_RAW),
        ("utf32 ascii", "hello wörld, café ✓ ".encode("utf-32-le"), MODE_UTF32),
        ("utf32 block", ("x" * 3000 + "日本語" * 40).encode("utf-32-le"), MODE_UTF32),
    ]
    fails = 0
    for label, data, mode in cases:
        if not data and label != "empty":
            continue
        dictionary = data[: min(len(data), 4000)] if label == "repeated text" else b""
        blob, orig_len, _, _ = pack(data, mode, dictionary, mem, sse, 11)
        back, _, _ = unpack(blob, dictionary)
        ok = back == data
        fails += 0 if ok else 1
        per = max(len(data) // (4 if mode == MODE_UTF32 else 1), 1)
        print(f"  {label:14s} {len(data):7,} B -> {len(blob):6,} B"
              f"   {8.0 * len(blob) / per:.3f} b/{'cp' if mode == MODE_UTF32 else 'byte'}"
              f"   {'ok' if ok else 'FAILED'}")
    return fails


def _bench(args, data, dictionary):
    mode = MODE_UTF32 if args.mode == "utf32" else MODE_RAW
    lr = args.lr or 11
    t0 = time.time()
    blob, orig_len, body_len, units = pack(
        data, mode, dictionary, args.mem, args.sse, lr
    )
    enc = time.time() - t0
    t0 = time.time()
    back, _, _ = unpack(blob, dictionary)
    dec = time.time() - t0
    per = units if mode == MODE_UTF32 else orig_len
    bpc = 8.0 * len(blob) / per if per else 0.0
    unit = "codepoint" if mode == MODE_UTF32 else "byte"
    print(
        f"in {orig_len:,} B ({units:,} {unit}s) -> {len(blob):,} B   "
        f"{bpc:.4f} b/{unit}   ratio {orig_len / max(len(blob), 1):.3f}:1"
    )
    print(
        f"model bytes {body_len:,}   enc {enc:.1f} s ({orig_len / max(enc, 1e-9) / 1e6:.2f} MB/s)"
        f"   dec {dec:.1f} s ({orig_len / max(dec, 1e-9) / 1e6:.2f} MB/s)"
    )
    if dictionary:
        print(f"dictionary {len(dictionary):,} B (primed, counted separately)")
    return back == data


def main():
    ap = argparse.ArgumentParser(description="WEZC prototype codec")
    ap.add_argument("cmd", choices=["compress", "decompress", "roundtrip", "bench", "selftest"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--mode", default="raw", choices=["raw", "utf32"])
    ap.add_argument("--limit", type=int, default=0, help="only process the first N bytes")
    ap.add_argument("--dict", dest="dictionary", help="dictionary/priming file")
    ap.add_argument("--mem", type=int, default=22)
    ap.add_argument("--sse", type=int, default=16)
    ap.add_argument("--lr", type=int, default=0, help="mixer learning-rate shift")
    a = ap.parse_args()

    if a.cmd == "selftest":
        bad = _selftest(min(a.mem, 18), min(a.sse, 14))
        print(f"selftest: {bad} failure(s)")
        return 1 if bad else 0
    if not a.file:
        raise SystemExit("this command needs an input file")

    data = open(a.file, "rb").read()
    if a.limit:
        data = data[: a.limit]
    dictionary = open(a.dictionary, "rb").read() if a.dictionary else b""

    if a.cmd == "compress":
        mode = MODE_UTF32 if a.mode == "utf32" else MODE_RAW
        t0 = time.time()
        blob, orig_len, _, units = pack(data, mode, dictionary, a.mem, a.sse, a.lr or 11)
        dt = time.time() - t0
        per = units if mode == MODE_UTF32 else orig_len
        print(
            f"{orig_len:,} B -> {len(blob):,} B   {8.0 * len(blob) / max(per, 1):.4f} bpc"
            f"   {orig_len / max(dt, 1e-9) / 1e6:.2f} MB/s"
        )
        if a.out:
            open(a.out, "wb").write(blob)
        return 0
    if a.cmd == "decompress":
        blob = data
        t0 = time.time()
        out, orig_len, _ = unpack(blob, dictionary)
        dt = time.time() - t0
        if len(out) != orig_len:
            print(f"length mismatch: got {len(out):,}, header says {orig_len:,}")
            return 1
        if a.out:
            open(a.out, "wb").write(out)
        print(f"{len(blob):,} B -> {orig_len:,} B   {orig_len / max(dt, 1e-9) / 1e6:.2f} MB/s")
        return 0
    if a.cmd == "roundtrip":
        mode = MODE_UTF32 if a.mode == "utf32" else MODE_RAW
        blob, orig_len, _, _ = pack(data, mode, dictionary, a.mem, a.sse, a.lr or 11)
        back, _, _ = unpack(blob, dictionary)
        ok = back == data
        print(f"round-trip {'OK' if ok else 'FAILED'}  ({orig_len:,} B)")
        return 0 if ok else 1
    return 0 if _bench(a, data, dictionary) else 1


if __name__ == "__main__":
    sys.exit(main())
