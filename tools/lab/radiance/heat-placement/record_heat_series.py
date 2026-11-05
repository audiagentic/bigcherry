#!/usr/bin/env python3
"""Record the live route-count file as a SERIES, for offline replay of the placement policy (r3_policy_replay).

A server started with --record-heat PREFIX rewrites PREFIX.rank0 (cumulative uint32 counts) every --record-period
seconds, by atomic rename, so only the last snapshot survives. This copies it to OUTDIR/NNNNNN.bin whenever its content
changed, until it is told to stop (SIGINT or SIGTERM) or --seconds have passed, so a run leaves the whole trajectory.

  record_heat_series.py --live PREFIX.rank0 --out DIR [--interval 1] [--seconds N]

Replay it with the component's r3_policy_replay (engines/radiance/components/moe-remote/worker_policy_replay_main.cpp).
"""
import argparse
import hashlib
import os
import signal
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, required=True, help="seconds between looks at the file")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long; 0 runs until signalled")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(1))
    last = None
    count = 0
    started = time.monotonic()
    while not stop and (not args.seconds or time.monotonic() - started < args.seconds):
        try:
            with open(args.live, "rb") as stream:
                data = stream.read()
        except FileNotFoundError:
            data = b""
        digest = hashlib.sha256(data).hexdigest() if data else None
        if data and digest != last:
            tmp = os.path.join(args.out, ".pending")
            with open(tmp, "wb") as out:
                out.write(data)
            os.replace(tmp, os.path.join(args.out, f"{count:06d}.bin"))
            count += 1
            last = digest
        time.sleep(args.interval)
    print(f"record_heat_series: {count} snapshot(s) in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
