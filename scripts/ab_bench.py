#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B of two ming-tts builds under the GPU lock.

    scripts/ab_bench.py BASE_BINARY CANDIDATE_BINARY [--rounds 5] [--weights DIR, or MING_WEIGHTS]

Each round runs `--bench --repeat 3` on A then B (order alternates per round) while holding the lock,
and keeps the warm runs (run 1 and 2). Prints the median steps/s, wall ms/step, first-audio ms and per-stage GPU ms,
each with its min-max range.
"""
import argparse, fcntl, os, re, statistics, subprocess

p = argparse.ArgumentParser()
p.add_argument("a"); p.add_argument("b")
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--weights", default=os.environ.get("MING_WEIGHTS"), required="MING_WEIGHTS" not in os.environ)
p.add_argument("--text", default=None)
args = p.parse_args()

RUN = re.compile(r"run (\d+): .*?([\d.]+) steps/s, first audio (\d+) ms")
STEP = re.compile(r"wall ([\d.]+)\s+gpu ([\d.]+)\s+\[llm ([\d.]+)\s+flow ([\d.]+)\s+agg ([\d.]+)\s+vae ([\d.]+)\]")
keys = ["steps/s", "first_ms", "wall_ms", "gpu_ms", "llm", "flow", "agg", "vae"]

def bench(binary):
    cmd = [binary, "--bench", "--repeat", "3", "--weights", args.weights] + (["--text", args.text] if args.text else [])
    with open("/tmp/ming-tts-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
    rows, lines = [], out.splitlines()
    for i, line in enumerate(lines):
        m = RUN.search(line)
        if m and int(m.group(1)) > 0:
            s = STEP.search(lines[i + 1])
            rows.append([float(m.group(2)), float(m.group(3))] + [float(x) for x in s.groups()])
    return rows

res = {"A": [], "B": []}
for r in range(args.rounds):
    order = [("A", args.a), ("B", args.b)] if r % 2 == 0 else [("B", args.b), ("A", args.a)]
    for name, binary in order:
        res[name] += bench(binary)
for name in "AB":
    cols = list(zip(*res[name]))
    print(name, " ".join(f"{k}={statistics.median(c):.2f}" for k, c in zip(keys, cols)), f"(n={len(res[name])})")
    print("  range", " ".join(f"{k}=[{min(c):.2f}-{max(c):.2f}]" for k, c in zip(keys, cols)))
a = statistics.median(r[0] for r in res["A"]); b = statistics.median(r[0] for r in res["B"])
print(f"B/A steps/s = {b / a:.3f}")
