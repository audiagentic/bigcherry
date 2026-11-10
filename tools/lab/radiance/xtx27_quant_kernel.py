"""VTP05: model-shaped IQ6R and W8A8 Radiance kernel screening, no model load.

Default --phase plan is card-free. --phase run --run acquires the global
exclusive GPU lease and runs ONE existing Radiance selftest performance case
with a matching prebuilt plugin. These modes are timing-only, NOT numeric
or model-quality validation. Never confuse IQ6R with GGUF Q6_K.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from xtx27_virtual_tp4 import CANDIDATES, parse_ratios, tile_counts

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path(os.environ.get("BC_WORK_ROOT", ROOT.parent)) / "xtx27-prefill" / "quant-shape"
INTERMEDIATE = 17408
HIDDEN = 5120
TILE = 256
CANDIDATE_NAMES = tuple(k for k, _ in CANDIDATES)
ARCH = {"xtx0": "gfx1100", "xtx1": "gfx1100", "r9700": "gfx1201"}


def geometry(candidate: str, gpu: str, projection: str, part: str = "physical") -> dict:
    if candidate not in CANDIDATE_NAMES or gpu not in ARCH:
        raise ValueError("unknown candidate or GPU role")
    if projection not in ("gate_up", "down") or part not in ("physical", "logical2", "logical3", "full"):
        raise ValueError("unknown projection or shard selection")
    if gpu != "r9700" and part.startswith("logical"):
        raise ValueError("only the R9700 has two logical shards")
    fractions = parse_ratios(dict(CANDIDATES)[candidate])
    widths = [v * TILE for v in tile_counts(fractions, width=INTERMEDIATE, tile=TILE)]
    if sum(widths) != INTERMEDIATE:
        raise ValueError("illegal total intermediate width")
    if part == "full":
        group = INTERMEDIATE
    elif gpu == "xtx0":
        group = widths[0]
    elif gpu == "xtx1":
        group = widths[1]
    elif part == "physical":
        group = widths[2] + widths[3]
    else:
        group = widths[2 if part == "logical2" else 3]
    if projection == "gate_up":
        n, k = 2 * group, HIDDEN
    else:
        n, k = HIDDEN, group
    if n % TILE or k % TILE:
        raise ValueError("IQR V4 kernel requires both N and K multiples of 256")
    return {
        "candidate": candidate, "fractions": [str(x) for x in fractions],
        "intermediate_shards": widths,
        "physical_weight_fractions": [round(widths[0] / INTERMEDIATE, 6),
                                      round(widths[1] / INTERMEDIATE, 6),
                                      round((widths[2] + widths[3]) / INTERMEDIATE, 6)],
        "gpu": gpu, "arch": ARCH[gpu], "part": part,
        "projection": projection, "N": n, "K": k, "effective_intermediate": group,
    }


def parse_devices(show: str) -> dict[str, int]:
    entries = re.findall(r"GPU\[(\d+)\].*?GFX Version:\s*(gfx\d+)", show)
    buckets: dict[str, list[int]] = {}
    for raw_index, arch in entries:
        buckets.setdefault(arch, []).append(int(raw_index))
    x = sorted(set(buckets.get("gfx1100", [])))
    r = sorted(set(buckets.get("gfx1201", [])))
    if len(x) != 2 or len(r) != 1:
        raise ValueError("device inventory must expose exactly two gfx1100 and one gfx1201")
    return {"xtx0": x[0], "xtx1": x[1], "r9700": r[0]}


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def selected_row(quant: str, m: int) -> str:
    if quant == "iq6r":
        return f"iqr k6 {'m64' if m <= 64 else 'tiled'}"
    if quant == "w8a8":
        return "gemm_i8a8_nt_m16" if m <= 16 else "gemm_i8a8_tiled"
    raise ValueError("quant must be iq6r or w8a8")


def extract_timing(stdout: str, *, quant: str, m: int) -> float:
    row = selected_row(quant, m)
    matches = re.findall(r"^\s*" + re.escape(row) + r"\s+(\d+(?:\.\d+)?)\s+us\b",
                         stdout, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"missing/ambiguous actual engine timing row: {row}")
    us = float(matches[0])
    if not 0 < us < 1e9 or re.search(r"(?im)^\s*FAIL\b", stdout):
        raise ValueError(f"invalid timing or selftest failure: {row}")
    return us


def build_paths(build: Path, gpu: str, quant: str) -> tuple[Path, Path]:
    exe = build / "bin" / "r4d_selftest"
    plugin = build / "radiance_home" / "kernels" / ("libr4d.so" if gpu == "r9700" else "libr3.so")
    if not exe.is_file() or not plugin.is_file():
        raise ValueError("matching prebuilt r4d_selftest and plugin required; no auto-build")
    executable_bytes, plugin_bytes = exe.read_bytes(), plugin.read_bytes()
    if quant == "iq6r":
        if b"--perfiqr" not in executable_bytes or any(
            row not in plugin_bytes
            for row in (b"gemm_iqr_nt_m64", b"gemm_iqr_tiled")
        ):
            raise ValueError("selected prebuilt binary/plugin lacks native IQR V4 timing rows")
    elif quant == "w8a8":
        if b"--perfw8" not in executable_bytes or any(
            row not in plugin_bytes
            for row in (b"gemm_i8a8_nt_m16", b"gemm_i8a8_tiled")
        ):
            raise ValueError("selected prebuilt binary/plugin lacks native W8A8 timing rows")
    else:
        raise ValueError("unknown quantization mode")
    return exe, plugin


def command(exe: Path, plugin: Path, *, quant: str, geo: dict, m: int,
            iters: int, cold_mib: int) -> list[str]:
    if m not in (1, 4, 8, 16, 64, 128, 512, 2048):
        raise ValueError("unsupported token-row count")
    if iters < 1 or iters > 20 or cold_mib not in (0, 64, 128, 256):
        raise ValueError("iters or cache-flush size out of bounds")
    args = [str(exe), str(plugin)]
    if quant == "iq6r":
        args += ["--perfiqr", str(geo["N"]), str(geo["K"]), str(m), "6", str(iters)]
    elif quant == "w8a8":
        args += ["--perfw8", str(geo["N"]), str(geo["K"]), str(m), str(iters)]
    else:
        raise ValueError("unsupported weight format")
    if cold_mib:
        args += ["--cold", str(cold_mib)]
    return args


def execute_one(args: argparse.Namespace, geo: dict) -> Path:
    # The parent acquire command sets this and holds the global GPU lease.
    if not args._under_lease or os.environ.get("VTP05_LEASED") != "1":
        raise ValueError("GPU execution only inside exclusive lease")
    build = args.build_dir.expanduser().resolve()
    exe, plugin = build_paths(build, args.gpu, args.quant)
    device_info = subprocess.run(["rocm-smi", "--showproductname"], text=True,
                                 capture_output=True, check=True, timeout=8)
    device_id = parse_devices(device_info.stdout)[args.gpu]
    cmd = command(exe, plugin, quant=args.quant, geo=geo, m=args.m,
                  iters=args.iters, cold_mib=args.cold_mib)
    out = args.scratch.expanduser().resolve() / "runs" / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" +
        args.quant + "-" + args.gpu
    )
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(device_id)
    env["ROCR_VISIBLE_DEVICES"] = str(device_id)
    receipt = {
        "status": "INCOMPLETE",
        "kind": "VTP05_RADIANCE_KERNEL_TIMING_ONLY_NOT_MODEL_QUALITY",
        "geometry": geo,
        "m": args.m, "quant": args.quant, "iters": args.iters,
        "cold_mib": args.cold_mib,
        "physical_device_index": device_id,
        "build_dir": str(build),
        "binary_sha256": sha256(exe), "plugin_sha256": sha256(plugin),
        "runner_sha256": sha256(Path(__file__)),
        "build_metadata_sha256": (
            sha256(build / "bigcherry-build-metadata-radiance.json")
            if (build / "bigcherry-build-metadata-radiance.json").is_file() else None
        ),
        "argv": cmd,
        "checkout": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True, check=True).stdout.strip(),
        "numerical_oracle": "NOT_TESTED_SELFTEST_PERF_MODE_TIMING_ONLY",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        run = subprocess.run(cmd, env=env, text=True, capture_output=True,
                             timeout=args.timeout_s, check=False)
        (out / "stdout.log").write_text(run.stdout)
        (out / "stderr.log").write_text(run.stderr)
        receipt["returncode"] = run.returncode
        receipt["selected_kernel"] = selected_row(args.quant, args.m)
        if run.returncode:
            receipt["status"] = "FAILED_SELFTEST"
        else:
            try:
                receipt["device_us"] = extract_timing(run.stdout, quant=args.quant, m=args.m)
                receipt["status"] = "TIMING_OBTAINED_NUMERICAL_UNCHECKED"
            except ValueError as exc:
                receipt["status"] = "FAILED_NO_SELECTED_KERNEL_TIMING"
                receipt["error"] = str(exc)
    except subprocess.TimeoutExpired as exc:
        receipt["status"] = "TIMEOUT"
        (out / "stdout.log").write_text(exc.stdout.decode(errors="replace") if
                                        isinstance(exc.stdout, bytes) else (exc.stdout or ""))
        (out / "stderr.log").write_text(exc.stderr.decode(errors="replace") if
                                        isinstance(exc.stderr, bytes) else (exc.stderr or ""))
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["status"] != "TIMING_OBTAINED_NUMERICAL_UNCHECKED":
        raise RuntimeError(f"failed/unqualified case; see {out / 'receipt.json'}")
    return out / "receipt.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "run"), default="plan")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--_under-lease", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--candidate", choices=CANDIDATE_NAMES, default="capacity")
    parser.add_argument("--gpu", choices=tuple(ARCH), default="r9700")
    parser.add_argument("--part", choices=("physical", "logical2", "logical3", "full"), default="physical")
    parser.add_argument("--projection", choices=("gate_up", "down"), default="gate_up")
    parser.add_argument("--quant", choices=("iq6r", "w8a8"), default="iq6r")
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--cold-mib", type=int, choices=(0, 64, 128, 256), default=256)
    parser.add_argument("--timeout-s", type=int, default=9)
    args = parser.parse_args(argv)
    try:
        geo = geometry(args.candidate, args.gpu, args.projection, args.part)
        checked = command(Path("selftest"), Path("matching-plugin"), quant=args.quant,
                          geo=geo, m=args.m, iters=args.iters, cold_mib=args.cold_mib)
    except ValueError as exc:
        parser.error(str(exc))
    if args.phase == "plan":
        print(json.dumps({
            "status": "CARD_FREE_PLAN_NO_GPU",
            "geometry": geo,
            "logical_intermediate_shards_all_candidates": {
                name: [v * TILE for v in tile_counts(parse_ratios(fractions),
                        width=INTERMEDIATE, tile=TILE)]
                for name, fractions in CANDIDATES
            },
            "per_case_command_template": checked,
            "selected_expected_kernel": selected_row(args.quant, args.m),
            "timing_only": True, "does_not_prove": [
                "Q6/IQ6R model quality", "GFLOPS proportional to end-to-end tokens/sec",
                "three-GPU parallelism", "full-layer/attention/GDN tensor mapping",
            ],
        }, indent=2))
        return 0
    if not args.run or args.build_dir is None:
        parser.error("--phase run requires --run and explicit --build-dir")
    if args.timeout_s < 1 or args.timeout_s > 30:
        parser.error("--timeout-s outside 1..30")
    scratch = args.scratch.expanduser().resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        parser.error("scratch must be outside repository")
    try:
        build_paths(args.build_dir.expanduser().resolve(), args.gpu, args.quant)
    except ValueError as exc:
        parser.error(str(exc))
    if not args._under_lease:
        orig = list(sys.argv[1:] if argv is None else argv)
        cmd = ["agent-lease", "acquire", "gpu", "--ttl", "5m",
               "--purpose", "VTP05 model-shape quantised native GEMM probe", "--",
               "env", "VTP05_LEASED=1", sys.executable, str(Path(__file__).resolve()),
               *orig, "--_under-lease"]
        return subprocess.run(cmd, check=False).returncode
    try:
        receipt = execute_one(args, geo)
        print(json.dumps({"status": "TIMING_ONLY_NUMERIC_UNCHECKED", "receipt": str(receipt)}, indent=2))
        return 0
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"VTP05 CASE FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
