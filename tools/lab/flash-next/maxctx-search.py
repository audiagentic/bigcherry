#!/usr/bin/env python3
"""Adaptive max-context search for the Flash-Next tensor split (XTX0, XTX1, R9700; MTP draft on the 6900).

For one KV type pair and ubatch: binary-search the context size; at each size, start from the split that
last worked (or the given start) and, on failure, shift split share away from the device the server log
reports as out of memory, up to MAX_TRIES times. A config counts only if it completes a ~10K-token request
(the profile client prefills and decodes), so request-time compute-buffer OOMs count as failures.

Usage: maxctx-search.py <llama-server> <out-root> <ctk> <ctv> <ub> [lo_ctx hi_ctx]
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(HERE, "long-ctx-profile.sh")
STEP = 8192
MAX_TRIES = 5
SHIFT = 0.04  # share moved away from the OOM device per retry


def run(bin_, out, ts, ctx, ctk, ctv, ub):
    env = dict(os.environ, TS=",".join(f"{w:.3f}" for w in ts), CTX=str(ctx), CTK=ctk, CTV=ctv, UB=str(ub),
               DEPTH="8192")
    r = subprocess.run(["bash", PROFILE, bin_, out, "timing"], env=env, capture_output=True, text=True)
    ok = any(l.startswith("timing: prompt") for l in r.stdout.splitlines())
    log = ""
    try:
        log = open(os.path.join(out, "timing.server.log"), errors="replace").read()
    except OSError:
        pass
    m = re.findall(r"allocating [\d.]+ MiB on device (\d+): cudaMalloc failed", log)
    oom_dev = int(m[-1]) if m else None
    vram = [float(x) / 1e9 for x in re.findall(r"VRAM Total Used Memory \(B\): (\d+)", r.stdout)]
    timing = [l for l in r.stdout.splitlines() if l.startswith("timing:")]
    return ok, oom_dev, vram, timing


def shift(ts, dev):
    if dev is None or dev > 2:
        return None
    w = list(ts)
    take = min(SHIFT, w[dev] * 0.5)
    w[dev] -= take
    others = [i for i in range(3) if i != dev]
    for i in others:
        w[i] += take / len(others)
    return w


def try_ctx(bin_, root, ts0, ctx, ctk, ctv, ub):
    ts = list(ts0)
    for attempt in range(MAX_TRIES):
        out = os.path.join(root, f"c{ctx}-t{attempt}")
        ok, dev, vram, timing = run(bin_, out, ts, ctx, ctk, ctv, ub)
        tag = ",".join(f"{w:.3f}" for w in ts)
        if ok:
            print(f"  ctx {ctx} ts {tag}: OK  vram {vram}  {' | '.join(timing)}", flush=True)
            return ts
        print(f"  ctx {ctx} ts {tag}: FAIL (oom device {dev})", flush=True)
        ts = shift(ts, dev)
        if ts is None:
            return None
    return None


def main():
    bin_, root, ctk, ctv, ub = sys.argv[1:6]
    lo = int(sys.argv[6]) if len(sys.argv) > 6 else 98304
    hi = int(sys.argv[7]) if len(sys.argv) > 7 else 262144
    os.makedirs(root, exist_ok=True)
    print(f"== K {ctk} V {ctv} ub {ub}: search {lo}..{hi}", flush=True)
    best_ts = [0.30, 0.30, 0.40]
    good = try_ctx(bin_, root, best_ts, lo, ctk, ctv, ub)
    if good is None:
        print(f"== K {ctk} V {ctv} ub {ub}: even {lo} fails", flush=True)
        return
    best_ctx, best_ts = lo, good
    lo_ok, hi_bad = lo, hi + STEP
    while hi_bad - lo_ok > STEP:
        mid = (lo_ok + hi_bad) // 2 // STEP * STEP
        if mid <= lo_ok:
            break
        r = try_ctx(bin_, root, best_ts, mid, ctk, ctv, ub)
        if r is None:
            hi_bad = mid
        else:
            lo_ok, best_ctx, best_ts = mid, mid, r
    print(f"== RESULT K {ctk} V {ctv} ub {ub}: max ctx {best_ctx} with -ts "
          f"{','.join(f'{w:.3f}' for w in best_ts)}", flush=True)


if __name__ == "__main__":
    main()
