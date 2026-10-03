"""Balanced multi-arm llama-bench comparison on one GPU.

Arm order rotates every round (round r runs arms starting at offset r), so
every arm occupies every position equally when rounds is a multiple of the
arm count. Every row records round and position so drift can be tested.

    python3 run_arms.py run --arm U10901=<bin> --arm U11126=<bin> ... \
        --model <gguf> --device 0 --rounds 8 --out results.jsonl
    python3 run_arms.py analyse results.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _run(args: argparse.Namespace) -> int:
    arms = [tuple(spec.split("=", 1)) for spec in args.arm]
    if args.rounds % len(arms):
        print("rounds must be a multiple of the arm count", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    env = dict(os.environ, ROCR_VISIBLE_DEVICES=str(args.device), HIP_VISIBLE_DEVICES="0")
    with out.open("w", encoding="utf-8") as sink:
        for rnd in range(args.rounds):
            order = arms[rnd % len(arms):] + arms[: rnd % len(arms)]
            for position, (name, binary) in enumerate(order):
                cmd = [binary, "-m", args.model, "-p", str(args.prompt), "-n", str(args.gen),
                       "-r", str(args.reps), "-ngl", "99", "-o", "json"]
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    print(f"{name} failed: {proc.stderr[-2000:]}", file=sys.stderr)
                    return 1
                start = proc.stdout.find("[")
                for row in json.loads(proc.stdout[start:]):
                    test = f"pp{row['n_prompt']}" if row["n_prompt"] else f"tg{row['n_gen']}"
                    sink.write(json.dumps({
                        "round": rnd, "position": position, "arm": name, "test": test,
                        "avg_ts": row["avg_ts"], "samples_ts": row.get("samples_ts"),
                        "build_commit": row.get("build_commit"), "gpu_info": row.get("gpu_info"),
                        "binary": binary,
                    }) + "\n")
                sink.flush()
                print(f"round {rnd} pos {position} {name} ok", flush=True)
                time.sleep(args.settle)
    return 0


def _mann_whitney_p(a: list[float], b: list[float]) -> float:
    """Exact two-sided Mann-Whitney U p-value by enumeration (small n)."""
    def u(x, y):
        return sum((xi > yj) + 0.5 * (xi == yj) for xi in x for yj in y)
    observed = u(a, b)
    pooled = a + b
    n = len(a)
    extreme = total = 0
    center = len(a) * len(b) / 2
    for combo in itertools.combinations(range(len(pooled)), n):
        chosen = [pooled[i] for i in combo]
        rest = [pooled[i] for i in range(len(pooled)) if i not in combo]
        total += 1
        if abs(u(chosen, rest) - center) >= abs(observed - center) - 1e-9:
            extreme += 1
    return extreme / total


def _analyse(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(args.results).read_text(encoding="utf-8").splitlines() if line]
    tests = sorted({r["test"] for r in rows})
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    for test in tests:
        print(f"== {test}")
        values = {a: [r["avg_ts"] for r in rows if r["arm"] == a and r["test"] == test] for a in arms}
        for a in arms:
            v = values[a]
            print(f"  {a:10} n={len(v)} mean={statistics.mean(v):9.2f} "
                  f"min={min(v):9.2f} max={max(v):9.2f} commit={rows[[r['arm'] for r in rows].index(a)]['build_commit']}")
        positions = sorted({r["position"] for r in rows})
        pos_means = [statistics.mean(r["avg_ts"] / statistics.mean(values[r["arm"]])
                                     for r in rows if r["test"] == test and r["position"] == p)
                     for p in positions]
        print("  position effect (mean / arm-mean): " + " ".join(f"p{p}={m:.4f}" for p, m in zip(positions, pos_means)))
        for base, cand in args.pair:
            a, b = values[base], values[cand]
            delta = (statistics.mean(b) / statistics.mean(a) - 1) * 100
            separated = max(b) < min(a) or min(b) > max(a)
            p = _mann_whitney_p(b, a) if len(a) + len(b) <= 20 else float("nan")
            print(f"  {cand} vs {base}: {delta:+.2f}%  complete_separation={separated}  MW p={p:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arm", action="append", required=True, help="NAME=path/to/llama-bench")
    run.add_argument("--model", required=True)
    run.add_argument("--device", type=int, required=True)
    run.add_argument("--rounds", type=int, required=True)
    run.add_argument("--reps", type=int, default=3)
    run.add_argument("--prompt", type=int, default=512)
    run.add_argument("--gen", type=int, default=128)
    run.add_argument("--settle", type=float, default=5.0)
    run.add_argument("--out", required=True)
    ana = sub.add_parser("analyse")
    ana.add_argument("results")
    ana.add_argument("--pair", nargs=2, action="append", default=[], metavar=("BASE", "CANDIDATE"))
    args = parser.parse_args()
    return _run(args) if args.cmd == "run" else _analyse(args)


if __name__ == "__main__":
    raise SystemExit(main())
