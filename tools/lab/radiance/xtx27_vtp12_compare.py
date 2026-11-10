"""VTP12 read-only post-analysis of paired FP8/INT8 rows in VTP05 receipts.

The historical --perfw8 selftest already timed both formats at each N/K/M.
Never launches GPUs, reads model weights, or claims model/collective throughput.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROWS = (
    "gemm_fp8a8_tiled",
    "gemm_i8a8_tiled",
    "gemm_fp8a8_nt_m16",
    "gemm_i8a8_nt_m16",
)
ARCH = {"xtx0": "gfx1100", "xtx1": "gfx1100", "r9700": "gfx1201"}
HEADER = re.compile(r"^8-bit trunk GEMMs\s+N=(\d+) K=(\d+) M=(\d+)\s+(\d+) iters$", re.MULTILINE)
TIMING = re.compile(r"^\s+(gemm_(?:fp8a8|i8a8)_(?:tiled|nt_m16))\s+(\d+(?:\.\d+)?) us\s+", re.MULTILINE)


def parse_perf(text: str, n: int, k: int, m: int, iters: int) -> dict[str, float]:
    headers = HEADER.findall(text)
    if len(headers) != 1 or tuple(map(int, headers[0])) != (n, k, m, iters):
        raise ValueError("missing/ambiguous --perfw8 header or geometry mismatch")
    pairs = TIMING.findall(text)
    times = {}
    for row, raw in pairs:
        if row in times:
            raise ValueError("duplicate kernel timing")
        times[row] = float(raw)
    required = ROWS if m <= 64 else ROWS[:2]
    if not set(required) <= times.keys() or any(
        not math.isfinite(x) or x <= 0 for x in times.values()
    ):
        raise ValueError("missing/invalid FP8 or INT8 native row")
    if re.search(r"(?m)^\s*(FAIL|--perfw8:|.*rc=-?\d+ \(shape not served\))", text):
        raise ValueError("selftest failure or unserved shape")
    return times


def read_pair(directory: Path) -> dict:
    receipt = json.loads((directory / "receipt.json").read_text())
    geo = receipt["geometry"]
    m, iters = receipt["m"], receipt["iters"]
    n, k = geo["N"], geo["K"]
    gpu = geo["gpu"]
    if (receipt.get("status") != "TIMING_OBTAINED_NUMERICAL_UNCHECKED"
            or receipt.get("kind") != "VTP05_RADIANCE_KERNEL_TIMING_ONLY_NOT_MODEL_QUALITY"
            or receipt.get("quant") != "w8a8" or receipt.get("returncode") != 0
            or receipt.get("cold_mib") != 0
            or receipt.get("cache_mode") != "warm_or_resident"
            or gpu not in ARCH or geo.get("arch") != ARCH[gpu]
            or geo.get("part") != "physical"
            or geo.get("projection") not in ("gate_up", "down")
            or not isinstance(m, int) or m < 1
            or not isinstance(iters, int) or iters < 1
            or receipt.get("observed_nkm") != [n, k, m]
            or not all(isinstance(v, int) and v > 0 for v in (n, k))
            or receipt.get("numerical_oracle") != "NOT_TESTED_SELFTEST_PERF_MODE_TIMING_ONLY"):
        raise ValueError(f"invalid/unqualified prior timing receipt: {directory}")
    argv = receipt.get("argv")
    if (not isinstance(argv, list) or len(argv) != 7
            or Path(argv[0]).name != "r4d_selftest"
            or Path(argv[1]).name not in ("libr3.so", "libr4d.so")
            or (Path(argv[1]).name == "libr4d.so") != (gpu == "r9700")
            or argv[2:] != ["--perfw8", str(n), str(k), str(m), str(iters)]
            or not all(isinstance(receipt.get(x), str) and len(receipt[x]) == 64
                       for x in ("binary_sha256", "plugin_sha256"))):
        raise ValueError(f"binary/ABI or --perfw8 command mismatch: {directory}")
    times = parse_perf((directory / "stdout.log").read_text(), n, k, m, iters)
    expected = "gemm_i8a8_nt_m16" if m <= 16 else "gemm_i8a8_tiled"
    if (receipt.get("selected_kernel") != expected
            or abs(times[expected] - receipt["device_us"]) > 1e-6):
        raise ValueError(f"original selected INT8 result does not match log: {directory}")
    return {
        "id": directory.name, "candidate": geo["candidate"], "m": m,
        "gpu": gpu, "projection": geo["projection"], "n": n, "k": k,
        "iters": iters, "selected_i8_row": expected,
        "selected_i8_us": times[expected],
        "fp8_preferred_row": "gemm_fp8a8_nt_m16" if m <= 64 else "gemm_fp8a8_tiled",
        "fp8_preferred_us": times["gemm_fp8a8_nt_m16" if m <= 64 else "gemm_fp8a8_tiled"],
        "best_observed_i8_us": min(times[r] for r in times if r.startswith("gemm_i8a8")),
        "best_observed_fp8_us": min(times[r] for r in times if r.startswith("gemm_fp8a8")),
        "all_rows_us": times,
        "binary_sha256": receipt["binary_sha256"],
        "plugin_sha256": receipt["plugin_sha256"],
        "checkout": receipt.get("checkout"),
    }


def barrier(cases: list[dict], *, mixed: bool, best_i8: bool = False) -> dict:
    by = {(x["gpu"], x["projection"]): x for x in cases}
    def cost(gpu: str, proj: str) -> float:
        x = by[(gpu, proj)]
        if mixed and gpu == "r9700":
            return x["fp8_preferred_us"]
        return x["best_observed_i8_us"] if best_i8 else x["selected_i8_us"]
    phases = []
    for proj in ("gate_up", "down"):
        times = {g: cost(g, proj) for g in ARCH}
        bottleneck = max(times, key=times.get)
        phases.append({"projection": proj, "critical_gpu": bottleneck,
                       "barrier_us": round(times[bottleneck], 4), "gpu_us": times})
    return {"sum_stage_barriers_us": round(sum(x["barrier_us"] for x in phases), 4),
            "phases": phases}


def analyze(runs_dir: Path) -> dict:
    if not runs_dir.is_dir():
        raise ValueError("missing VTP05 run directory")
    rows = [read_pair(p) for p in sorted(runs_dir.iterdir())
            if p.is_dir() and (p / "receipt.json").exists()]
    if not rows:
        raise ValueError("no source receipts")
    by_key = {}
    for x in rows:
        key = (x["candidate"], x["m"], x["gpu"], x["projection"])
        if key in by_key:
            raise ValueError(f"duplicate source experiment case: {key}")
        by_key[key] = x
    if len({x["binary_sha256"] for x in rows}) != 1 or len({x["checkout"] for x in rows}) != 1:
        raise ValueError("historical cases use different binary/checkout")
    for gpu in ARCH:
        if len({x["plugin_sha256"] for x in rows if x["gpu"] == gpu}) > 1:
            raise ValueError("historical cases use different plugin for same GPU")
    groups = sorted({(x["candidate"], x["m"]) for x in rows})
    summaries = []
    for candidate, m in groups:
        cases = [x for x in rows if (x["candidate"], x["m"]) == (candidate, m)]
        if {(x["gpu"], x["projection"]) for x in cases} != {
            (g, proj) for g in ARCH for proj in ("gate_up", "down")
        }:
            continue  # partial independent tier: retain rows, do not extrapolate.
        images = {(x["gpu"], x["projection"]): x for x in cases}
        for gpu in ARCH:
            gpu_rows = [images[gpu, p] for p in ("gate_up", "down")]
            if len({x["plugin_sha256"] for x in gpu_rows}) != 1:
                raise ValueError("different plugin within a physical GPU pair")
        baseline = barrier(cases, mixed=False)
        mixed = barrier(cases, mixed=True)
        summaries.append({
            "candidate": candidate, "m": m,
            "uniform_i8_selected": baseline,
            "mixed_xtx_i8_r9700_fp8_provisional": mixed,
            "uniform_i8_best_observed_local_rows": barrier(cases, mixed=False, best_i8=True),
            "ideal_stage_barrier_saving_percent": round(
                100 * (1 - mixed["sum_stage_barriers_us"] / baseline["sum_stage_barriers_us"]), 4
            ),
            "same_invocation_r9700_gate_down": {
                "i8_selected_us": round(sum(images["r9700", p]["selected_i8_us"]
                                          for p in ("gate_up", "down")), 4),
                "fp8_preferred_us": round(sum(images["r9700", p]["fp8_preferred_us"]
                                            for p in ("gate_up", "down")), 4),
            },
        })
    return {
        "kind": "VTP12_REANALYSIS_PRIOR_VTP05_TIMINGS_NO_NEW_GPU_RUN",
        "precision": "synthetic constant weights and activations; warm-only",
        "numeric": "selftest perf mode unchecked; NOT model quality",
        "fp8_row_selection": "provisional valid row; not proven selected in engine graph",
        "simultaneous_three_gpu_execution": False,
        "activation_quantization_included": False,
        "collective_included": False,
        "source_receipt_count": len(rows),
        "source_ids": [x["id"] for x in rows],
        "summaries": summaries,
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, help="optional JSON output outside checkout")
    args = p.parse_args()
    report = analyze(args.runs_dir)
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        root = Path(__file__).resolve().parents[3]
        if args.output.resolve().is_relative_to(root):
            p.error("report may not be written into checkout")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
