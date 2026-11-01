#!/usr/bin/env python3
"""RHP05 stage A, recommend-only: watch a recording Flash-Next server's live route histogram and say when its --heat file
no longer fits the traffic.

The launcher started with --record-heat PREFIX rewrites PREFIX.rank0 (cumulative counts) every --record-period seconds.
Every --every seconds this takes the counts of the last --window seconds (the newest snapshot minus the one from a window
ago, so old traffic does not dilute a change), runs heat-drift.py on them against the active --heat file, and when a
re-rank is advised writes the re-ranked hot-first file to --out and prints the restart command. It never restarts the
server itself: a restart drains every session, so applying it stays an operator decision.

  heat-watch.py --heat ACTIVE.hot-first --live PREFIX.rank0 --out NEW.hot-first [--every 300] [--window 1800]
                [--tail 5931] [--heads 18640] [--min-share 2.5] [--ratio 1.5] [--once]
"""
import argparse
import array
import collections
import os
import subprocess
import sys
import tempfile
import time

DRIFT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heat-drift.py")
SIZE = 128 * 512


def read(path: str):
    a = array.array("I")
    try:
        with open(path, "rb") as f:
            a.frombytes(f.read())
    except OSError:
        return None
    return a if len(a) == SIZE else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heat", required=True, help="the server's active --heat file (hot-first)")
    ap.add_argument("--live", required=True, help="the live histogram the server rewrites (PREFIX.rank0)")
    ap.add_argument("--out", required=True, help="where a recommended re-ranked hot-first file goes")
    ap.add_argument("--every", type=float, default=300.0, help="seconds between checks")
    ap.add_argument("--window", type=float, default=1800.0, help="seconds of traffic a check judges")
    ap.add_argument("--tail", type=int, default=5931, help="experts the tail owns (flashnext-4gpu's 14 GiB tail)")
    ap.add_argument("--heads", type=int, default=18640, help="experts the heads own together")
    ap.add_argument("--min-share", type=float, default=2.5)
    ap.add_argument("--ratio", type=float, default=1.5)
    ap.add_argument("--once", action="store_true", help="one check over the whole histogram, then exit with heat-drift's code")
    a = ap.parse_args()
    snaps = collections.deque()   # (time, counts)
    advised = False
    while True:
        now = time.time()
        cur = read(a.live)
        if cur is not None:
            snaps.append((now, cur))
            while len(snaps) > 1 and now - snaps[1][0] >= a.window:
                snaps.popleft()
            base = snaps[0][1] if not a.once and len(snaps) > 1 else array.array("I", [0] * SIZE)
            window = array.array("I", (max(0, x - y) for x, y in zip(cur, base)))
            if any(window):
                with tempfile.NamedTemporaryFile(suffix=".rank0", delete=False) as t:
                    t.write(window.tobytes())
                r = subprocess.run([sys.executable, DRIFT, a.heat, t.name, "--tail", str(a.tail), "--heads", str(a.heads),
                                    "--min-share", str(a.min_share),
                                    "--ratio", str(a.ratio), "--write", a.out], capture_output=True, text=True)
                os.unlink(t.name)
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                lines = [l for l in r.stdout.splitlines() if l.startswith(("active file", "re-ranked"))]
                if r.returncode == 3:
                    print(f"{stamp} RE-RANK ADVISED: {'; '.join(lines)}\n  restart with: "
                          f"bash tools/lab/radiance/launch-flashnext-best.sh stop && "
                          f"bash tools/lab/radiance/launch-flashnext-best.sh start --heat {a.out} [your other arguments]", flush=True)
                    advised = True
                elif r.returncode == 0:
                    print(f"{stamp} keep: {'; '.join(lines)}", flush=True)
                    advised = False
                else:
                    print(f"{stamp} heat-drift failed: {r.stderr.strip()}", file=sys.stderr, flush=True)
                if a.once:
                    return r.returncode
        elif a.once:
            print(f"heat-watch: no readable histogram at {a.live}", file=sys.stderr)
            return 1
        time.sleep(a.every)
    return 3 if advised else 0


if __name__ == "__main__":
    sys.exit(main())
