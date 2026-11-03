#!/usr/bin/env python3
# Copyright 2026 Codevar Project
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# https://www.apache.org/licenses/LICENSE-2.0
#
"""Benchmark the WEZC prototype against brotli and the usual baselines.

Brotli is the comparison that matters for Wredit: it is a general purpose
compressor with a fixed bitstream and a built-in 122 KiB static dictionary, so
any ratio WEZC claims has to be paid for with modelling rather than with a
slower decoder. Every row reports size, bits per unit, encode and decode
throughput and peak RSS, because a live text buffer cares about all four - a
codec that wins 5% of ratio and loses 100x of decode speed is not a win.

Usage:
  wezc_bench.py CORPUS [--sizes 100000,1000000] [--units bytes|codepoints]
                       [--wezc] [--mem 22] [--sse 16] [--lr 11] [--csv OUT]
  wezc_bench.py CORPUS --blocks 12 --dict DICT [--wezc]

`--wezc` runs the prototype itself, which is pure Python and costs about a
minute per megabyte; the reference rows always run. `--blocks` compares 4 KiB
blocks, the Wredit store's unit, with each codec's own dictionary mechanism.
"""

import argparse
import contextlib
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time

# (name, compress argv, decompress argv, how the codec is invoked)
# "file": the codec writes to {o}; "stdout": its stdout is the archive.
REFERENCE = [
    ("gzip -9", ["gzip", "-9", "-c", "{i}"], ["gzip", "-d", "-c", "{i}"], "stdout"),
    ("brotli q0", ["brotli", "-f", "-q", "0", "-o", "{o}", "{i}"],
     ["brotli", "-f", "-d", "-o", "{o}", "{i}"], "file"),
    ("brotli q1", ["brotli", "-f", "-q", "1", "-o", "{o}", "{i}"],
     ["brotli", "-f", "-d", "-o", "{o}", "{i}"], "file"),
    ("brotli q4", ["brotli", "-f", "-q", "4", "-o", "{o}", "{i}"],
     ["brotli", "-f", "-d", "-o", "{o}", "{i}"], "file"),
    ("brotli q6", ["brotli", "-f", "-q", "6", "-o", "{o}", "{i}"],
     ["brotli", "-f", "-d", "-o", "{o}", "{i}"], "file"),
    ("brotli q9", ["brotli", "-f", "-q", "9", "-o", "{o}", "{i}"],
     ["brotli", "-f", "-d", "-o", "{o}", "{i}"], "file"),
    ("brotli q11", ["brotli", "-f", "-q", "11", "-o", "{o}", "{i}"],
     ["brotli", "-f", "-d", "-o", "{o}", "{i}"], "file"),
    ("brotli q11 w24", ["brotli", "-f", "-q", "11", "-w", "24", "-o", "{o}", "{i}"],
     ["brotli", "-f", "-d", "-o", "{o}", "{i}"], "file"),
    ("zstd -1", ["zstd", "-q", "-1", "-f", "-o", "{o}", "{i}"],
     ["zstd", "-q", "-d", "-f", "-o", "{o}", "{i}"], "file"),
    ("zstd -19", ["zstd", "-q", "-19", "-f", "-o", "{o}", "{i}"],
     ["zstd", "-q", "-d", "-f", "-o", "{o}", "{i}"], "file"),
    ("xz -9e", ["xz", "-9e", "-c", "{i}"], ["xz", "-d", "-c", "{i}"], "stdout"),
]

BLOCK_SIZE = 4096


def self_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run(argv, redirect_to=None):
    """Run argv, timing it and sampling its peak RSS through /usr/bin/time."""
    if not shutil.which(argv[0]):
        return None
    with tempfile.NamedTemporaryFile("w+", delete=False) as tag:
        name = tag.name
    cmd = ["/usr/bin/time", "-f", "%e %M", "-o", name] + argv
    sink_cm = (open(redirect_to, "wb") if redirect_to
               else contextlib.nullcontext(subprocess.DEVNULL))
    t0 = time.perf_counter()
    try:
        with sink_cm as sink:
            proc = subprocess.run(cmd, stdout=sink, stderr=subprocess.DEVNULL,
                                  stdin=subprocess.DEVNULL, check=False)
    except OSError:
        return None
    wall = time.perf_counter() - t0
    try:
        parts = open(name).read().split()
    except OSError:
        parts = []
    os.unlink(name)
    if proc.returncode != 0 or len(parts) < 2:
        return None
    return wall, int(parts[1]) / 1024.0


def bench_codec(name, enc, dec, how, data, workdir, per_unit):
    """Compress, decompress and verify one reference codec."""
    src = os.path.join(workdir, "in.bin")
    packed = os.path.join(workdir, "out.z")
    restored = os.path.join(workdir, "out.bin")
    with open(src, "wb") as fh:
        fh.write(data)
    enc_argv = [a.replace("{i}", src).replace("{o}", packed) for a in enc]
    if how == "stdout":
        got = run(enc_argv, redirect_to=packed)
    else:
        if os.path.exists(packed):
            os.unlink(packed)
        got = run(enc_argv)
    if got is None:
        print(f"{name}: skipped, command failed or missing", file=sys.stderr)
        return None
    enc_s, enc_rss = got
    enc_s = net_time(enc_s, spawn_cost(workdir))
    blob = open(packed, "rb").read()
    # decompression always reads the archive, whatever the codec's convention
    dec_argv = [a.replace("{i}", packed).replace("{o}", restored) for a in dec]
    if os.path.exists(restored):
        os.unlink(restored)
    if how == "stdout":
        got = run(dec_argv, redirect_to=restored)
    else:
        got = run(dec_argv)
    if got is None:
        print(f"{name}: skipped, decompress failed", file=sys.stderr)
        return None
    dec_s, dec_rss = got
    dec_s = net_time(dec_s, spawn_cost(workdir))
    back = open(restored, "rb").read()
    for path in (packed, restored):
        if os.path.exists(path):
            os.unlink(path)
    return {
        "codec": name,
        "bytes": len(blob),
        "bits_per_unit": 8.0 * len(blob) / per_unit,
        "encode_MBps": len(data) / max(enc_s, 1e-9) / 1e6,
        "decode_MBps": len(data) / max(dec_s, 1e-9) / 1e6,
        "enc_RSS_MiB": round(enc_rss, 1),
        "dec_RSS_MiB": round(dec_rss, 1),
        "verified": back == data,
    }


def bench_wezc(wezc, data, mode, dictionary, mem, sse, lr, per_unit, label=None):
    """Encode and decode with the prototype, in this process."""
    t0 = time.perf_counter()
    blob, orig_len, _, _ = wezc.pack(data, mode, dictionary, mem, sse, lr)
    enc_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    back, _, _ = wezc.unpack(blob, dictionary)
    dec_s = time.perf_counter() - t0
    rss = self_rss_mib()
    return {
        "codec": label or f"WEZC m{mem} s{sse}",
        "bytes": len(blob),
        "bits_per_unit": 8.0 * len(blob) / per_unit,
        "encode_MBps": orig_len / max(enc_s, 1e-9) / 1e6,
        "decode_MBps": orig_len / max(dec_s, 1e-9) / 1e6,
        "enc_RSS_MiB": round(rss, 1),
        "dec_RSS_MiB": round(rss, 1),
        "verified": back == data,
    }


def load_wezc():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wezc.py")
    spec = importlib.util.spec_from_file_location("wezc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COLUMNS = ("codec", "bytes", "bits/unit", "ratio", "enc MB/s", "dec MB/s",
           "enc RSS", "dec RSS", "ok")


def report_header(unit, stream=sys.stdout):
    cols = list(COLUMNS)
    cols[2] = f"bits/{unit}"
    print(f"{cols[0]:16s} {cols[1]:>11s} {cols[2]:>10s} {cols[3]:>8s} "
          f"{cols[4]:>9s} {cols[5]:>9s} {cols[6]:>8s} {cols[7]:>8s}  {cols[8]}",
          file=stream)
    print("-" * 96, file=stream)
    stream.flush()


def report_row(r, stream=sys.stdout):
    ratio = 8.0 / r["bits_per_unit"] if r["bits_per_unit"] else 0.0
    print(f"{r['codec']:16s} {r['bytes']:11,} {r['bits_per_unit']:10.4f} "
          f"{ratio:7.2f}:1 {r['encode_MBps']:9.3g} {r['decode_MBps']:9.3g} "
          f"{r['enc_RSS_MiB']:8.1f} {r['dec_RSS_MiB']:8.1f}  "
          f"{'yes' if r['verified'] else 'NO'}", file=stream)
    stream.flush()


_SPAWN = None


def net_time(raw, spawn):
    """Strip the CLI startup from a measured run, floored at the resolution.

    `/usr/bin/time` and process spawn are worth a couple of milliseconds; a run
    that finishes inside that window is simply too fast to resolve, and the
    floor keeps the table from claiming infinite throughput.
    """
    floor = max(spawn * 0.05, 1e-9)
    return max(raw - spawn, floor)


def spawn_cost(workdir):
    """Wall time of a brotli run that does no real work, memoised.

    A 4 KiB block compresses in well under a millisecond, so the process spawn
    and the /usr/bin/time wrapper would otherwise dominate the per-block
    throughput and make the CLI codecs look slower than in-process ones.
    """
    global _SPAWN
    if _SPAWN is not None:
        return _SPAWN
    src = os.path.join(workdir, "spawn.bin")
    packed = os.path.join(workdir, "spawn.br")
    with open(src, "wb") as fh:
        fh.write(b"\0" * 16)
    samples = []
    for _ in range(5):
        got = run(["brotli", "-f", "-q", "0", "-o", packed, src])
        if got is None:
            _SPAWN = 0.0
            return 0.0
        samples.append(got[0])
        os.unlink(packed)
    samples.sort()
    _SPAWN = samples[len(samples) // 2]
    return _SPAWN


def block_test(wezc, args, workdir, dictionary):
    """4 KiB blocks: the Wredit store unit, each codec with its own dictionary."""
    if not shutil.which("brotli"):
        raise SystemExit("brotli not found in PATH (need the CLI)")
    data = open(args.corpus, "rb").read(args.blocks * BLOCK_SIZE)
    blocks = [data[i:i + BLOCK_SIZE] for i in range(0, args.blocks * BLOCK_SIZE, BLOCK_SIZE)
              if len(data[i:i + BLOCK_SIZE]) == BLOCK_SIZE]
    if not blocks:
        raise SystemExit("the corpus is too short for that many blocks")
    src = os.path.join(workdir, "blk.bin")
    packed = os.path.join(workdir, "blk.br")
    out_blk = os.path.join(workdir, "blk.out")
    dict_argv = ["-D", args.dictionary] if args.dictionary else []
    # brotli needs the same raw dictionary on both sides of the stream
    configs = [
        ("brotli q9 +dict", ["-q", "9"] + dict_argv, ["-d"] + dict_argv),
        ("brotli q11 +dict", ["-q", "11"] + dict_argv, ["-d"] + dict_argv),
        ("brotli q11", ["-q", "11"], ["-d"]),
        ("brotli q9", ["-q", "9"], ["-d"]),
    ]
    n = len(blocks) * BLOCK_SIZE
    spawn = spawn_cost(workdir)
    print(f"\n=== {len(blocks)} blocks of {BLOCK_SIZE} B from {args.corpus}"
          f"  (dictionary {len(dictionary):,} B)")
    print(f"    brotli throughput is net of {spawn * 1e3:.2f} ms of process spawn")
    report_header("byte")
    rows = []
    for name, enc_flags, dec_flags in configs:
        if "+dict" in name and not args.dictionary:
            continue
        total, enc_s, dec_s = 0, 0.0, 0.0
        enc_rss, dec_rss = 0.0, 0.0
        ok = True
        for b in blocks:
            with open(src, "wb") as fh:
                fh.write(b)
            got = run(["brotli", "-f"] + enc_flags + ["-o", packed, src])
            if got is None:
                raise SystemExit(f"{name}: compress failed on {src}")
            enc_s += net_time(got[0], spawn)
            enc_rss = max(enc_rss, got[1])
            total += os.path.getsize(packed)
            got = run(["brotli", "-f"] + dec_flags + ["-o", out_blk, packed])
            if got is None:
                raise SystemExit(f"{name}: decompress failed on {packed}")
            dec_s += net_time(got[0], spawn)
            dec_rss = max(dec_rss, got[1])
            ok = ok and open(out_blk, "rb").read() == b
            os.unlink(packed)
        rows.append({"codec": name, "bytes": total, "bits_per_unit": 8.0 * total / n,
                     "encode_MBps": n / max(enc_s, 1e-9) / 1e6,
                     "decode_MBps": n / max(dec_s, 1e-9) / 1e6,
                     "enc_RSS_MiB": round(enc_rss, 1),
                     "dec_RSS_MiB": round(dec_rss, 1), "verified": ok})
        report_row(rows[-1])
    if args.wezc:
        for use_dict, label in ((True, "WEZC +priming"), (False, "WEZC no dict")):
            if not args.dictionary and use_dict:
                continue
            dict_arg = dictionary if use_dict else b""
            t0 = time.perf_counter()
            blobs = [wezc.encode_stream(b, dict_arg, args.mem, args.sse, args.lr)
                     for b in blocks]
            enc_s = time.perf_counter() - t0
            total = sum(len(b) for b in blobs)
            t0 = time.perf_counter()
            back = [wezc.decode_stream(blob, len(src_blk), dict_arg,
                                       args.mem, args.sse, args.lr)
                    for blob, src_blk in zip(blobs, blocks)]
            dec_s = time.perf_counter() - t0
            rss = round(self_rss_mib(), 1)
            rows.append({"codec": label, "bytes": total,
                         "bits_per_unit": 8.0 * total / n,
                         "encode_MBps": n / max(enc_s, 1e-9) / 1e6,
                         "decode_MBps": n / max(dec_s, 1e-9) / 1e6,
                         "enc_RSS_MiB": rss, "dec_RSS_MiB": rss,
                         "verified": back == blocks})
            report_row(rows[-1])
    return rows


def main():
    ap = argparse.ArgumentParser(description="WEZC versus brotli benchmark")
    ap.add_argument("corpus")
    ap.add_argument("--sizes", default="100000,1000000")
    ap.add_argument("--units", default="bytes", choices=["bytes", "codepoints"])
    ap.add_argument("--wezc", action="store_true")
    ap.add_argument("--mem", type=int, default=22)
    ap.add_argument("--sse", type=int, default=16)
    ap.add_argument("--lr", type=int, default=11)
    ap.add_argument("--dict", dest="dictionary")
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--csv")
    a = ap.parse_args()

    dictionary = open(a.dictionary, "rb").read() if a.dictionary else b""
    wezc = load_wezc()
    rows_all = []
    with tempfile.TemporaryDirectory() as workdir:
        if a.blocks:
            rows = block_test(wezc, a, workdir, dictionary)
            rows_all += [dict(r, size=a.blocks * BLOCK_SIZE) for r in rows]
        else:
            for size in [int(s) for s in a.sizes.split(",") if s]:
                chunk = open(a.corpus, "rb").read(size)
                units = len(chunk) // 4 if a.units == "codepoints" else len(chunk)
                unit = "cp" if a.units == "codepoints" else "byte"
                print(f"\n=== {os.path.basename(a.corpus)} slice {len(chunk):,} B"
                      f"  ({units:,} {a.units})")
                report_header(unit)
                rows = []
                for name, enc, dec, how in REFERENCE:
                    r = bench_codec(name, enc, dec, how, chunk, workdir, units)
                    if r:
                        rows.append(r)
                        report_row(r)
                if a.wezc:
                    mode = wezc.MODE_UTF32 if a.units == "codepoints" else wezc.MODE_RAW
                    r = bench_wezc(wezc, chunk, mode, dictionary, a.mem, a.sse,
                                   a.lr, units)
                    rows.append(r)
                    report_row(r)
                rows_all += [dict(r, size=len(chunk)) for r in rows]
        if a.csv:
            with open(a.csv, "w") as fh:
                json.dump(rows_all, fh, indent=1)
    failed = [r["codec"] for r in rows_all if not r["verified"]]
    if failed:
        print(f"\nround-trip FAILED for: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
