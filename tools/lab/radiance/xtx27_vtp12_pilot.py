"""VTP12 bounded, exclusive-lease synthetic INT8/FP8 split ABBA pilot.

Reuses VTP05's measured Radiance --perfw8 selftest rows and precision
parser; starts one isolated physical GPU subprocess at a time. No model
loading, no simultaneous TP, no collectives, no quality claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import xtx27_quant_kernel as q
import xtx27_vtp12_compare as compare
from xtx27_virtual_tp4 import parse_ratios, tile_counts

LEASE_PURPOSE = "VTP12 gfx11-gfx12 Qwen27B mixed-format FFN ABBA pilot"
CANDIDATES = {
    "control": ("0.30", "0.30", "0.20", "0.20"),
    "r9700-43": ("0.28676470588235294", "0.28676470588235294",
                 "0.21323529411764706", "0.21323529411764706"),
    "r9700-43-skew": ("0.29411764705882353", "0.27941176470588235",
                      "0.21323529411764706", "0.21323529411764706"),
    "r9700-45": ("0.275", "0.275", "0.225", "0.225"),
    "r9700-50": ("0.25", "0.25", "0.25", "0.25"),
}
ROLES = ("xtx0", "xtx1", "r9700")
PROJECTIONS = ("gate_up", "down")
SCHEMA = "vtp12-ffn-synthetic-abba-v1"


def geometry(candidate: str, gpu: str, projection: str) -> dict:
    if candidate not in CANDIDATES or gpu not in ROLES or projection not in PROJECTIONS:
        raise ValueError("unsupported VTP12 candidate, GPU or projection")
    fr = parse_ratios(CANDIDATES[candidate])
    # Preserve 256-tile v1 control geometry for immutable replay.
    # The 43% candidate needs legal 128-wide K tiles for 4992/4992/7424.
    tile = 128 if candidate in ("r9700-43", "r9700-43-skew") else q.TILE
    widths = [t * tile for t in tile_counts(fr, width=q.INTERMEDIATE, tile=tile)]
    physical = (widths[0], widths[1], widths[2] + widths[3])
    width = physical[ROLES.index(gpu)]
    n, k = ((2 * width, q.HIDDEN) if projection == "gate_up"
            else (q.HIDDEN, width))
    if width <= 0 or n % 256 or k % 128:
        raise ValueError("illegal FFN matrix/tile dimensions")
    return {
        "candidate": candidate, "fractions": list(CANDIDATES[candidate]),
        "intermediate_shards": widths, "physical_intermediate_shards": list(physical),
        "physical_weight_fractions": [round(x / q.INTERMEDIATE, 6) for x in physical],
        "gpu": gpu, "arch": q.ARCH[gpu], "part": "physical",
        "projection": projection, "N": n, "K": k,
        "effective_intermediate": width,
    }


def schedule(challenger: str, m: int) -> list[dict]:
    if challenger not in ("r9700-43", "r9700-43-skew", "r9700-45", "r9700-50"):
        raise ValueError("challenger must be a heavier R9700 candidate")
    if m not in (1, 4, 8, 64):
        raise ValueError("pilot limited to M=1/4/8/64")
    result = []
    for gpu in ROLES:
        for proj in PROJECTIONS:
            for arm in ("control", challenger, challenger, "control"):
                result.append({"candidate": arm, "gpu": gpu, "projection": proj,
                               "m": m, "geometry": geometry(arm, gpu, proj)})
    assert len(result) == 24
    return result


def validate_context() -> None:
    if os.getenv("VTP12_LEASED") != "1":
        raise RuntimeError("VTP12 GPU runner requires lease wrapper")
    for key in q.MASK_VARIABLES:
        if key in os.environ:
            raise RuntimeError(f"inherited GPU device filter: {key}")
    lease = subprocess.run(["agent-lease", "status"], capture_output=True,
                           text=True, check=True, timeout=8).stdout
    if f'for "{LEASE_PURPOSE}"' not in lease:
        raise RuntimeError("global GPU lease not held for this VTP12 experiment")
    show = subprocess.run(["rocm-smi", "--showpids"], capture_output=True,
                          text=True, check=True, timeout=8).stdout
    if "No KFD PIDs currently running" not in show:
        raise RuntimeError("GPU processes already resident; fail closed")


def preflight(build: Path) -> dict:
    exe = build / "bin" / "r4d_selftest"
    if not exe.is_file():
        raise ValueError("missing built Radiance selftest")
    result = {
        "build_dir": str(build),
        "selftest_sha256": q.sha256(exe),
        "plugins": {},
        "runner_sha256": q.sha256(Path(__file__)),
        "vtp05_runner_sha256": q.sha256(Path(q.__file__)),
        "parser_sha256": q.sha256(Path(compare.__file__)),
    }
    for role in ROLES:
        e, plugin = q.build_paths(build, role, "w8a8")
        if e != exe or not q.contains_file_tokens(
            plugin, (b"gemm_fp8a8_tiled", b"gemm_fp8a8_nt_m16")
        ):
            raise ValueError("same-build FP8 and INT8 kernel rows unavailable")
        result["plugins"][role] = {"path": str(plugin), "sha256": q.sha256(plugin)}
    return result


def compile_manifest(build: Path, challenger: str, m: int, iters: int) -> dict:
    if iters < 3 or iters > 15:
        raise ValueError("pilot requires 3..15 warm timing iterations")
    return {
        "schema": SCHEMA,
        "kind": "VTP12_ONE_CARD_AT_A_TIME_SYNTHETIC_WARM_ABBA_NOT_SERVING",
        "created_utc": datetime.now(UTC).isoformat(),
        "challenger": challenger,
        "m": m,
        "iters": iters,
        "cases": schedule(challenger, m),
        "provenance": preflight(build),
    }


def to_summary(records: list[dict], *, challenger: str) -> dict:
    if len(records) != 24:
        raise ValueError("incomplete ABBA pilot")
    per_card = {}
    for gpu in ROLES:
        per_card[gpu] = {}
        for proj in PROJECTIONS:
            rows = [x for x in records if x["gpu"] == gpu and x["projection"] == proj]
            if [x["candidate"] for x in rows] != ["control", challenger, challenger, "control"]:
                raise ValueError("ABBA case order/completeness mismatch")
            def mean_arm(candidate: str, fmt: str, rows: list[dict] = rows) -> float:
                points = [x["fp8_us"] if fmt == "fp8" else x["i8_us"]
                          for x in rows if x["candidate"] == candidate]
                if len(points) != 2:
                    raise ValueError("missing paired replicate")
                return sum(points) / 2
            per_card[gpu][proj] = {
                "control_i8_us": round(mean_arm("control", "i8"), 4),
                "challenger_i8_us": round(mean_arm(challenger, "i8"), 4),
                "challenger_fp8_us": round(mean_arm(challenger, "fp8"), 4),
            }
    arms = {}
    for arm in ("control_i8_us", "challenger_i8_us", "challenger_fp8_us"):
        phases = []
        for proj in PROJECTIONS:
            times = {g: per_card[g][proj][
                "challenger_i8_us" if arm == "challenger_fp8_us" and g != "r9700" else arm
            ] for g in ROLES}
            biggest = max(times, key=times.get)
            phases.append({"projection": proj, "slowest": biggest,
                           "ideal_barrier_us": times[biggest], "per_gpu_us": times})
        arms[arm] = {
            "ideal_ffn_stage_barriers_us": round(sum(p["ideal_barrier_us"] for p in phases), 4),
            "phases": phases,
        }
    return {
        "status": "SYNTHETIC_WARM_ONLY_NUMERICAL_UNCHECKED",
        "per_card": per_card,
        "stage_estimates": arms,
        "warning": "Not simultaneous three-GPU execution; FP8-selected row is provisional, "
                   "no activation preparation/collective/attention/GDN or real-model quality",
    }


def run(manifest: dict, out_dir: Path, timeout: int) -> None:
    if timeout < 5 or timeout > 45:
        raise ValueError("per-case timeout must be 5..45 seconds")
    validate_context()
    manifest_path = out_dir / "campaign.json"
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    inventory = subprocess.run(["rocm-smi", "--showproductname"], text=True,
                               capture_output=True, check=True, timeout=8).stdout
    devices = q.parse_devices(inventory)
    (out_dir / "inventory.txt").write_text(inventory)
    results = []
    try:
        for i, case in enumerate(manifest["cases"]):
            lease = subprocess.run(["agent-lease", "status"], capture_output=True,
                                   text=True, check=True, timeout=8).stdout
            if f'for "{LEASE_PURPOSE}"' not in lease:
                raise RuntimeError("lease vanished while pilot ran")
            role = case["gpu"]
            exe = Path(manifest["provenance"]["build_dir"]) / "bin" / "r4d_selftest"
            plugin = Path(manifest["provenance"]["plugins"][role]["path"])
            if (q.sha256(exe) != manifest["provenance"]["selftest_sha256"]
                    or q.sha256(plugin) != manifest["provenance"]["plugins"][role]["sha256"]):
                raise RuntimeError("binary/plugin changed since preflight")
            argv = q.command(exe, plugin, quant="w8a8", geo=case["geometry"],
                             m=case["m"], iters=manifest["iters"], cold_mib=0)
            rec_path = out_dir / f"case-{i:02d}.json"
            stdout_path = out_dir / f"case-{i:02d}.stdout"
            stderr_path = out_dir / f"case-{i:02d}.stderr"
            record = {**case, "ordinal": i, "argv": argv,
                      "device_index": devices[role],
                      "status": "LAUNCHED", "timestamp_utc": datetime.now(UTC).isoformat()}
            rec_path.write_text(json.dumps(record, indent=2) + "\n")
            t0 = time.monotonic()
            env = q.isolated_device_env(os.environ.copy(), devices[role])
            cp = subprocess.run(argv, env=env, capture_output=True, text=True,
                                timeout=timeout, check=False)
            record["elapsed_wall_s"] = round(time.monotonic() - t0, 3)
            record["returncode"] = cp.returncode
            stdout_path.write_text(cp.stdout)
            stderr_path.write_text(cp.stderr)
            if cp.returncode != 0:
                raise ValueError("Radiance selftest failed")
            values = compare.parse_perf(cp.stdout, case["geometry"]["N"],
                                        case["geometry"]["K"], case["m"],
                                        manifest["iters"])
            row = "gemm_i8a8_nt_m16" if case["m"] <= 64 else "gemm_i8a8_tiled"
            frow = "gemm_fp8a8_nt_m16" if case["m"] <= 64 else "gemm_fp8a8_tiled"
            record.update({
                "status": "SYNTHETIC_WARM_ONLY_NUMERICAL_UNCHECKED",
                "i8_kernel": row, "fp8_kernel": frow,
                "i8_us": values[row], "fp8_us": values[frow],
                "all_rows_us": values,
                "stdout_sha256": hashlib.sha256(cp.stdout.encode()).hexdigest(),
            })
            rec_path.write_text(json.dumps(record, indent=2) + "\n")
            results.append(record)
        summary = to_summary(results, challenger=manifest["challenger"])
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"status": summary["status"],
                          "output": str(out_dir / "summary.json"),
                          "stage_estimates": summary["stage_estimates"]}, indent=2), flush=True)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        (out_dir / "failure.json").write_text(json.dumps(
            {"failed_at_case_index": len(results), "error": str(exc),
             "recorded_completed_cases": len(results)}, indent=2) + "\n")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "run"), default="plan")
    parser.add_argument("--challenger", choices=("r9700-43", "r9700-43-skew", "r9700-45", "r9700-50"), default="r9700-43")
    parser.add_argument("--m", type=int, choices=(1, 4, 8, 64), default=4)
    parser.add_argument("--iters", type=int, default=9)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--_under-lease", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase == "plan":
        print(json.dumps({"status": "CARD_FREE_NO_GPU",
                          "schedule": schedule(args.challenger, args.m),
                          "cases": 24, "iters": args.iters}, indent=2))
        return 0
    if not args.run or not args.build_dir or not args.output_dir:
        parser.error("run requires --run, --build-dir and --output-dir")
    target = args.output_dir.resolve()
    checkout = Path(__file__).resolve().parents[3]
    if target.is_relative_to(checkout) or target == checkout:
        parser.error("output directory cannot live in checkout")
    manifest = compile_manifest(args.build_dir.resolve(), args.challenger, args.m, args.iters)
    if not args._under_lease:
        original = sys.argv[1:]
        argv = ["agent-lease", "acquire", "gpu", "--ttl", "15m",
                "--purpose", LEASE_PURPOSE, "--",
                "env", "VTP12_LEASED=1", sys.executable, str(Path(__file__).resolve()),
                *original, "--_under-lease"]
        return subprocess.run(argv, check=False).returncode
    run(manifest, target, args.timeout_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
