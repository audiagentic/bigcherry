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
MAX_TRIES = 6
SHIFT = 0.02  # share moved away from the OOM device per retry
CAP_GB = [25.75, 25.75, 34.2]  # XTX0, XTX1, R9700 usable VRAM


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
    if oom_dev is None:  # e.g. CUBLAS_STATUS_ALLOC_FAILED creating a handle: the ROCm error names the device
        d = re.findall(r"current device: (\d+)", log)
        oom_dev = int(d[-1]) if d else None
    vram = [float(x) / 1e9 for x in re.findall(r"VRAM Total Used Memory \(B\): (\d+)", r.stdout)]
    timing = [l for l in r.stdout.splitlines() if l.startswith("timing:")]
    return ok, oom_dev, vram, timing


def shift(ts, dev, free):
    """Move SHIFT of split share off the OOM device onto the device with the most free VRAM last time
    a config loaded (falls back to the other two equally)."""
    if dev is None or dev > 2:
        return None
    w = list(ts)
    take = min(SHIFT, w[dev] * 0.5)
    w[dev] -= take
    others = [i for i in range(3) if i != dev]
    if free:
        w[max(others, key=lambda i: free[i])] += take
    else:
        for i in others:
            w[i] += take / len(others)
    return w


LAST_FREE = []


def try_ctx(bin_, root, ts0, ctx, ctk, ctv, ub):
    global LAST_FREE
    ts = list(ts0)
    for attempt in range(MAX_TRIES):
        out = os.path.join(root, f"c{ctx}-t{attempt}")
        ok, dev, vram, timing = run(bin_, out, ts, ctx, ctk, ctv, ub)
        tag = ",".join(f"{w:.3f}" for w in ts)
        if ok:
            if len(vram) >= 3:
                LAST_FREE = [CAP_GB[i] - vram[i] for i in range(3)]
            print(f"  ctx {ctx} ts {tag}: OK  vram {vram}  {' | '.join(timing)}", flush=True)
            return ts
        print(f"  ctx {ctx} ts {tag}: FAIL (oom device {dev})", flush=True)
        ts = shift(ts, dev, LAST_FREE)
        if ts is None:
            return None
    return None


def main():
    bin_, root, ctk, ctv, ub = sys.argv[1:6]
    lo = int(sys.argv[6]) if len(sys.argv) > 6 else 98304
    hi = int(sys.argv[7]) if len(sys.argv) > 7 else 262144
    os.makedirs(root, exist_ok=True)
    print(f"== K {ctk} V {ctv} ub {ub}: search {lo}..{hi}", flush=True)
    # XTX1 carries ~4 GB of unsplit data and, with the R9700, the two KV heads (XTX0 holds none), so XTX0 gets most.
    best_ts = [float(x) for x in os.environ.get("START_TS", "0.31,0.27,0.42").split(",")]
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
