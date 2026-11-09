"""Isolated dual-gfx1100 dense prefill experiment, using MEN03 engine-bench.

A measurement requires --run and an externally acquired exclusive GPU lease.
It does not build or patch Radiance. All outputs live outside the checkout.
TTFT-derived prefill is NOT GPU-only kernel throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from bigcherry.tuning import engine_bench

ROOT = Path(__file__).resolve().parents[3]
WIRES = frozenset({"exact", "wht6"})
BATCHES = frozenset({512, 1024, 2048, 4096, 8192})


@dataclass(frozen=True)
class Arm:
    wire: str
    batch: int

    @property
    def name(self) -> str:
        return f"{self.wire}-b{self.batch}"


def parse_arm(raw: str) -> Arm:
    match = re.fullmatch(r"(exact|wht6):([0-9]+)", raw)
    if not match or int(match[2]) not in BATCHES:
        raise ValueError(f"arm must be exact|wht6:512|1024|2048|4096|8192: {raw!r}")
    return Arm(match[1], int(match[2]))


def parse_devices(raw: str) -> tuple[int, int]:
    if not re.fullmatch(r"[0-9]+,[0-9]+", raw):
        raise ValueError("devices must be two comma-separated inventory indices")
    first, second = map(int, raw.split(","))
    if first == second:
        raise ValueError("two distinct GPUs required")
    return first, second


def args_for(arm: Arm, *, context: int, decode: int, profile: bool) -> list[str]:
    if context <= decode + 32:
        raise ValueError("context must leave room for the decode and prompt")
    result = [
        "--embedding-placement", "ram",
        "--tp", "2", "--tp-wire", arm.wire, "--p2p", "off",
        "--max-model-len", str(context), "--max-num-seqs", "1",
        "--max-num-batched-tokens", str(arm.batch),
        "--kv-cache-dtype", "fp8", "--num-speculative-tokens", "0",
    ]
    if profile:
        result.append("--profile-ops")
    return result


def checked_rows(record: dict, depths: tuple[int, ...], reps: int) -> dict[int, dict]:
    """Fail closed: never average failures, missing reps, or bogus telemetry."""
    if record.get("server_error") or not (record.get("shutdown") or {}).get("clean"):
        raise ValueError("server failed or did not stop cleanly")
    if not record.get("served_model"):
        raise ValueError("missing served model identity")
    rows = record.get("rows")
    if not isinstance(rows, list) or len(rows) != len(depths) * reps:
        raise ValueError("missing or extra measurement rows")
    by_depth: dict[int, dict] = {}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (row.get("depth"), row.get("rep"))
        if key in seen or key[0] not in depths or not isinstance(key[1], int) or not 0 <= key[1] < reps:
            raise ValueError("duplicated or unexpected depth/repetition")
        seen.add(key)
        if row.get("error") or any(
            not isinstance(row.get(field), (int, float))
            or not math.isfinite(row[field]) or row[field] <= 0
            for field in ("prompt_tokens", "completion_tokens", "ttft_s", "prefill_tps")
        ):
            raise ValueError(f"invalid measured request: {key}")
        by_depth.setdefault(key[0], {"prefill": [], "ttft": [], "prompt_tokens": [], "decode": []})
        group = by_depth[key[0]]
        group["prefill"].append(float(row["prefill_tps"]))
        group["ttft"].append(float(row["ttft_s"]))
        group["prompt_tokens"].append(int(row["prompt_tokens"]))
        dt = row.get("decode_tps")
        if isinstance(dt, (int, float)) and math.isfinite(dt) and dt > 0:
            group["decode"].append(float(dt))
    if len(seen) != len(depths) * reps:
        raise ValueError("unmeasured depth/rep")
    return {
        depth: {
            "prefill_tps_median": round(statistics.median(group["prefill"]), 2),
            "prefill_tps_range": [round(min(group["prefill"]), 2), round(max(group["prefill"]), 2)],
            "ttft_s_median": round(statistics.median(group["ttft"]), 4),
            "prompt_tokens_range": [min(group["prompt_tokens"]), max(group["prompt_tokens"])],
            "decode_tps_median": (
                round(statistics.median(group["decode"]), 2)
                if len(group["decode"]) == reps else None
            ),
            "requests": reps,
        }
        for depth, group in sorted(by_depth.items())
    }


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def provenance(args: argparse.Namespace, devices: tuple[int, int], arms: list[Arm]) -> dict:
    metadata = args.build / "bigcherry-build-metadata-radiance.json"
    return {
        "utc": datetime.now(UTC).isoformat(),
        "checkout_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "build": str(args.build.resolve()),
        "engine_sha256": sha256(args.build / "bin" / "radiance"),
        "build_metadata": json.loads(metadata.read_text()) if metadata.exists() else None,
        "model": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "corpus_sha256": sha256(args.corpus),
        "gpu_inventory_indices": list(devices),
        "arms": [asdict(a) for a in arms],
        "depths": list(args.depths),
        "context": args.context,
        "decode": args.decode,
        "reps": args.reps,
        "profile_ops": args.profile_ops,
        "metric": "MEN03 prompt_tokens / HTTP time-to-first-text-token; not kernel-only prefill",
        "limitations": [
            "Measurements include tokenization, scheduling, graph/cold-start and HTTP.",
            "Unique prompt nonces avoid prefix hits, but exact token counts can differ by request.",
            "wht6 may be lossy; these timings do not establish output quality.",
            "Strata Flash-Next MoE is not a matched dense-27B comparison.",
        ],
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="actually launch GPU serving; otherwise print the plan")
    parser.add_argument("--build", type=Path, required=True, help="completed Radiance build with bin/ and radiance_home/")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new scratch directory; never a repo directory")
    parser.add_argument("--devices", required=True, help="two inventory GPU indices, e.g. 0,1")
    parser.add_argument("--arm", action="append", default=[], help="e.g. exact:2048 or wht6:4096")
    parser.add_argument("--depths", type=int, nargs="+", default=[512, 2048, 6144])
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--decode", type=int, default=64)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--port", type=int, default=18411)
    parser.add_argument("--profile-ops", action="store_true", help="separate diagnostic; profiler changes timings")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        devices = parse_devices(args.devices)
        arms = [parse_arm(raw) for raw in (args.arm or ["exact:2048", "wht6:2048"])]
        if len(set(arms)) != len(arms):
            raise ValueError("duplicate arm")
        if args.reps < 1 or args.reps > 12 or args.decode < 2 or args.decode > 512:
            raise ValueError("reps must be 1..12, decode 2..512")
        if not args.depths or any(d < 64 or d * 1.15 + args.decode + 16 >= args.context for d in args.depths):
            raise ValueError("depths must be >=64 and fit context with a tokenization margin")
        if len(set(args.depths)) != len(args.depths):
            raise ValueError("duplicate depth")
        if not (18400 <= args.port <= 18419):
            raise ValueError("port must be in the private lab range 18400..18419")
        for item in (args.build / "bin" / "radiance", args.model, args.corpus):
            if not item.is_file():
                raise ValueError(f"file unavailable: {item}")
        if not (args.build / "radiance_home").is_dir():
            raise ValueError("build has no radiance_home")
        if args.output.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("output must be outside the repository")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("output directory is not empty; never overwrite measurements")
        for arm in arms:
            args_for(arm, context=args.context, decode=args.decode, profile=args.profile_ops)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    record = provenance(args, devices, arms)
    print(json.dumps({"measurement_plan": record}, indent=2), flush=True)
    if not args.run:
        return 0

    args.output.mkdir(parents=True, exist_ok=False) if not args.output.exists() else None
    (args.output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    summary = {"provenance": record, "results": {}, "failures": []}
    try:
        for arm in arms:
            print(f"ARM_START {arm.name}", flush=True)
            try:
                result = engine_bench.run(
                    engine="radiance", binary=args.build / "bin" / "radiance",
                    model=str(args.model), out_dir=args.output,
                    depths=args.depths, corpus_path=args.corpus,
                    reps=args.reps, decode=args.decode, label=arm.name, port=args.port,
                    shutdown_method="sigint", health_timeout_s=900,
                    env={
                        "HIP_VISIBLE_DEVICES": ",".join(map(str, devices)),
                        "ROCR_VISIBLE_DEVICES": ",".join(map(str, devices)),
                    },
                    extra_args=args_for(arm, context=args.context, decode=args.decode,
                                        profile=args.profile_ops),
                    report=lambda line, name=arm.name: print(f"{name}: {line}", flush=True),
                )
                summary["results"][arm.name] = checked_rows(result, tuple(args.depths), args.reps)
                print(f"ARM_OK {arm.name}", flush=True)
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                summary["failures"].append({"arm": arm.name, "error": str(exc)})
                print(f"ARM_FAILED {arm.name}: {exc}", flush=True)
                break  # A failure may mean VRAM or lifetime trouble; stop before another GPU run.
    finally:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
