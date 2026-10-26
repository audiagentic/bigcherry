#!/usr/bin/env python3
"""RAD20: compare the main and burst arms of a rad20-burst-profile.sh run.

Per arm: helper kernel durations by name (rocprofv3 kernel trace, both helpers), the launch gaps between consecutive
kernels on one queue, R3_MOE_WORKER_TRACE per-request times (helper GPU time, engine wait) inside the decode window,
and the helper GPUs' shader clock while busy.  Usage: rad20-profile-analyse.py <run dir>
"""
import csv
import glob
import statistics as st
import sys
from pathlib import Path


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1)))] if v else float("nan")


def summary(v):
    return f"n {len(v):7d}  mean {st.mean(v):8.2f}  p50 {pct(v, .5):8.2f}  p95 {pct(v, .95):8.2f}" if v else "n 0"


def kernels(arm_dir):
    by_name, gaps = {}, []
    for f in glob.glob(str(arm_dir / "trace*" / "**" / "*kernel_trace.csv"), recursive=True):
        rows = list(csv.DictReader(open(f)))
        last_end = {}
        for r in sorted(rows, key=lambda r: int(r["Start_Timestamp"])):
            s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
            name = r["Kernel_Name"].split("(")[0][:60]
            by_name.setdefault(name, []).append((e - s) / 1000.0)
            q = r.get("Queue_Id") or r.get("Stream_Id") or "0"
            if q in last_end and 0 <= s - last_end[q] < 50_000:   # gaps within a request (< 50 us)
                gaps.append((s - last_end[q]) / 1000.0)
            last_end[q] = e
    return by_name, gaps


def traces(arm_dir, t0, t1):
    out = {"helper_gpu_us": [], "wait_us": [], "local_us": []}
    for f in arm_dir.glob("helper*.log"):
        for line in open(f, errors="replace"):
            if not line.startswith("TRACE,"):
                continue
            p = line.strip().split(",")
            kv = dict(zip(p[1::2], p[2::2]))
            if not (t0 <= int(kv["wall_ns"]) / 1e9 <= t1) or int(kv["rows"]) > 16:
                continue
            for k in out:
                out[k].append(float(kv[k]))
    return out


def clocks(arm_dir, t0, t1):
    res = {}
    for r in csv.DictReader(open(arm_dir / "clocks.csv")):
        if t0 <= float(r["t"]) <= t1 and r["busy_pct"] and int(r["busy_pct"]) > 0:
            res.setdefault(r["bdf"], []).append(float(r["sclk_mhz"]))
    return res


def main():
    run = Path(sys.argv[1])
    arms = {}
    for arm in ("main", "burst"):
        d = run / arm
        t0, t1 = (float(x) for x in open(d / "window.txt").read().split())
        arms[arm] = (kernels(d), traces(d, t0, t1), clocks(d, t0, t1), open(d / "tokens.txt").read().split())
    for arm, (_, _, _, tok) in arms.items():
        print(f"{arm}: completion tokens {tok}")
    print("\n== per-request (small lane, decode window) ==")
    for k in ("helper_gpu_us", "wait_us", "local_us"):
        for arm in arms:
            print(f"{k:14s} {arm:5s} {summary(arms[arm][1][k])}")
    print("\n== helper shader clock while busy (MHz) ==")
    for arm in arms:
        for bdf, v in sorted(arms[arm][2].items()):
            print(f"{arm:5s} {bdf}  {summary(v)}")
    print("\n== kernel launch gaps on one queue (us, < 50 us) ==")
    for arm in arms:
        print(f"{arm:5s} {summary(arms[arm][0][1])}")
    print("\n== kernels by total time (us): main vs burst ==")
    km, kb = arms["main"][0][0], arms["burst"][0][0]
    for name in sorted(km, key=lambda n: -sum(km[n]))[:20]:
        a, b = km[name], kb.get(name, [])
        if not b:
            continue
        d = 100.0 * (st.mean(b) - st.mean(a)) / st.mean(a)
        print(f"{name:60s} n {len(a):7d}/{len(b):7d}  mean {st.mean(a):8.2f} -> {st.mean(b):8.2f}  {d:+6.2f}%  "
              f"p50 {pct(a, .5):7.2f} -> {pct(b, .5):7.2f}")


if __name__ == "__main__":
    main()
