"""Adaptive IQ6R checkpoint conversion; run only after micro controller handoff.

One, two or four new GPTQ tensors per complete RAD2. A separate controller
cannot start while the one-tensor controller holds its advisory lock. Every
stage is sealed and hashed before any predecessor can be reused.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from iq6r_v4_controller import WORK

# The calibrated quantizer helper needs the experiment-local NumPy installation.
sys.path.insert(0, str(WORK / "pydeps"))
from iq6r_v4_convert_experimental import HESSIANS, PINNED, ROOT, managed_build, sha
from iq6r_v4_micro_convert import MICRO_ROOT, ORIGINAL_STAGE_ONE, stage_one_record
from iq6r_v4_stage_recipes import recipes
from iq6r_v4_staged_convert import (
    STATUS,
    call,
    desired_encoding_counts,
    runner_command,
    verify_stage_info,
    write_json,
)

DIR = ROOT / "adaptive-conversion"
STATE = ROOT / "conversion-controller-adaptive.json"
LOCK = ROOT / "conversion-controller-adaptive.lock"
MICRO_LOCK = ROOT / "conversion-controller-micro.lock"
TOTAL = 292
MIN_FREE = 85 * 1024**3


def source_recipes() -> list[bytes]:
    return recipes(
        (PINNED / "data/recipes/q38-27b-iq6r-v4.recipe").read_bytes(),
        (PINNED / "data/recipes/q38-27b-w8a8-df2.recipe").read_bytes(),
        chunk=1,
    )


def choose_batch(previous: list[dict]) -> int:
    """Start conservatively. Increase only on actual fast sealed stage timings."""
    if not previous:
        return 1
    latest = previous[-1]
    if latest["elapsed_s"] > 1800:
        return 1
    if len(previous) >= 2 and all(
        r["elapsed_s"] <= 600 for r in previous[-2:]
    ):
        return 4
    if latest["elapsed_s"] <= 900:
        return 2
    return 1


def micro_checkpoint(sources: list[bytes]) -> tuple[Path, dict]:
    """Inherit only the latest real micro checkpoint, or original Stage 1."""
    original = stage_one_record(sources[0])
    receipts = sorted(MICRO_ROOT.glob("micro-*.json")) if MICRO_ROOT.is_dir() else []
    numbers = [int(p.stem[6:]) for p in receipts
               if re.fullmatch(r"micro-\d{4}", p.stem)]
    if numbers != list(range(2, 2 + len(numbers))):
        raise RuntimeError("micro receipts missing or out of sequence")
    if not numbers:
        return ORIGINAL_STAGE_ONE, {
            "sha256": original["sha256"], "bytes": original["bytes"],
            "gptq_cumulative": 4, "stage": 1,
        }
    num = numbers[-1]
    path = MICRO_ROOT / f"micro-{num:04d}.rad"
    if not path.is_file():
        raise RuntimeError("latest sealed micro artifact is absent")
    record = json.loads((MICRO_ROOT / f"micro-{num:04d}.json").read_text())
    if (record.get("stage") != num or record.get("gptq_cumulative") != num + 3
            or record.get("recipe_sha256") != hashlib.sha256(sources[num - 1]).hexdigest()
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != sha(path)):
        raise RuntimeError("latest micro checkpoint identity mismatches")
    return path, record


def adaptive_history(sources: list[bytes], initial_sha: str, initial_count: int) -> list[dict]:
    paths = sorted(DIR.glob("adaptive-*.json")) if DIR.is_dir() else []
    history: list[dict] = []
    before_sha, before_count = initial_sha, initial_count
    for i, path in enumerate(paths, 1):
        if not re.fullmatch(r"adaptive-\d{4}", path.stem) or path.name != f"adaptive-{i:04d}.json":
            raise RuntimeError("unexpected adaptive checkpoint receipt")
        record = json.loads(path.read_text())
        count = record.get("gptq_cumulative")
        if (record.get("stage") != i or not isinstance(count, int)
                or not before_count < count <= min(before_count + 4, TOTAL)
                or record.get("predecessor_sha256") != before_sha
                or record.get("recipe_sha256") != hashlib.sha256(sources[count - 4]).hexdigest()):
            raise RuntimeError(f"adaptive stage {i} receipt identity mismatch")
        history.append(record)
        before_sha, before_count = record["sha256"], count
    if history:
        last = history[-1]
        artifact = DIR / f"adaptive-{len(history):04d}.rad"
        if (not artifact.is_file() or artifact.stat().st_size != last["bytes"]
                or sha(artifact) != last["sha256"]):
            raise RuntimeError("latest sealed adaptive RAD2 is missing or modified")
    return history


def free_space_guard(path: Path) -> None:
    usage = os.statvfs(path)
    available = usage.f_bavail * usage.f_frsize
    if available < MIN_FREE:
        raise RuntimeError(f"disk free-space guard: {available} < {MIN_FREE}")


def next_job(sources: list[bytes]) -> dict:
    base, base_record = micro_checkpoint(sources)
    history = adaptive_history(sources, base_record["sha256"],
                               base_record["gptq_cumulative"])
    if history:
        last = history[-1]
        base = DIR / f"adaptive-{len(history):04d}.rad"
        base_record = last
    before = base_record["gptq_cumulative"]
    selected = choose_batch(history)
    # On a clean handoff, use the measured micro-stage runtime rather than
    # discarding that performance evidence and restarting at batch=1.
    # Always begin the larger-batch schedule at 2, never jump straight to 4.
    if not history and before > 4:
        last_micro = json.loads(
            (MICRO_ROOT / f"micro-{before - 3:04d}.json").read_text()
        )
        if last_micro.get("elapsed_s", 10801) <= 900:
            selected = 2
    batch = min(selected, TOTAL - before)
    count = before + batch
    return {
        "stage": len(history) + 1,
        "gptq_before": before,
        "gptq_after": count,
        "batch": batch,
        "previous_artifact": str(base),
        "previous_sha256": base_record["sha256"],
        "recipe_sha256": hashlib.sha256(sources[count - 4]).hexdigest()
        if batch else None,
        "done": before == TOTAL,
    }


def convert(job: dict, sources: list[bytes], timeout: int) -> dict:
    if job["done"]:
        raise ValueError("all GPTQ tensors already sealed")
    lease = subprocess.run(["agent-lease", "status"], capture_output=True, text=True, check=True).stdout
    if "gpu: held" not in lease or "IQ6R" not in lease:
        raise RuntimeError("exclusive IQ6R lease required")
    build, manifest = managed_build()
    original_status = json.loads(STATUS.read_text())
    if (manifest["tag"] != original_status["release"]
            or manifest["binary_sha256"] != original_status["binary_sha256"]
            or sha(build / "bin/rad-convert") != original_status["converter_sha256"]):
        raise RuntimeError("managed converter identity differs from Stage 1")
    DIR.mkdir(parents=True, exist_ok=True)
    free_space_guard(DIR)
    idx = job["stage"]
    prefix = DIR / f"adaptive-{idx:04d}"
    complete = prefix.with_suffix(".rad")
    pending = DIR / f"adaptive-{idx:04d}.incomplete.rad"
    receipt = prefix.with_suffix(".json")
    recipe = prefix.with_suffix(".recipe")
    if any(p.exists() for p in (complete, pending, receipt, recipe)):
        raise RuntimeError("refusing to overwrite existing adaptive stage")
    source = sources[job["gptq_after"] - 4]
    recipe.write_bytes(source)
    env = os.environ.copy()
    env["CALIB"] = str(HESSIANS)
    env["HIP_VISIBLE_DEVICES"] = "2"
    env.pop("ROCR_VISIBLE_DEVICES", None)
    input_rad = Path(job["previous_artifact"])
    plan = runner_command(build, recipe, Path("/dev/null"), input_rad, plan_only=True)
    if call(plan, env, DIR / f"adaptive-{idx:04d}-plan-out.log",
            DIR / f"adaptive-{idx:04d}-plan-err.log", 180):
        raise RuntimeError("adaptive plan-only validation failed")
    start = time.monotonic()
    command = runner_command(build, recipe, pending, input_rad, plan_only=False)
    rc = call(command, env, DIR / f"adaptive-{idx:04d}-stdout.log",
              DIR / f"adaptive-{idx:04d}-stderr.log", timeout)
    if rc:
        raise RuntimeError(f"adaptive stage {idx} convert failed exit {rc}, no retry")
    if not pending.is_file() or pending.stat().st_size < 5 * 1024**3:
        raise RuntimeError("adaptive output missing or truncated")
    info_result = subprocess.run([str(build / "bin/rad-info"), str(pending)],
                                 capture_output=True, text=True, check=False, timeout=300)
    (DIR / f"adaptive-{idx:04d}-radinfo.txt").write_text(
        info_result.stdout + info_result.stderr)
    if info_result.returncode:
        raise RuntimeError("adaptive RAD2 parse failed")
    info = verify_stage_info(info_result.stdout, source)
    if sum(desired_encoding_counts(source).values()) != job["gptq_after"]:
        raise RuntimeError("adaptive checkpoint counts do not match recipe")
    record = {
        **job, "sha256": sha(pending), "bytes": pending.stat().st_size,
        "elapsed_s": round(time.monotonic() - start, 2),
        "radinfo": info, "production_promotion": False,
        "complete_model_qualified": False,
    }
    pending.rename(complete)
    write_json(receipt, record)
    # Keep the last two sealed adaptive artifacts; never modify Stage 1 or
    # the inherited micro artifact.
    if idx >= 3:
        (DIR / f"adaptive-{idx-2:04d}.rad").unlink(missing_ok=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--managed", action="store_true")
    parser.add_argument("--stage-timeout", type=int, default=10800)
    args = parser.parse_args()
    if not 300 <= args.stage_timeout <= 10800:
        parser.error("stage-timeout must be 300..10800")
    if args.managed:
        token = os.environ.get("BC_IQ6R_ADAPTIVE_TOKEN", "")
        state = json.loads(STATE.read_text())
        if (not token or hashlib.sha256(token.encode()).hexdigest() !=
                state.get("controller_token_sha256") or
                state.get("pid") == os.getpid()):
            raise RuntimeError("managed worker missing controller authorization")
        try:
            os.kill(int(state["pid"]), 0)
        except (ProcessLookupError, KeyError, ValueError) as exc:
            raise RuntimeError("adaptive controller is no longer live") from exc
        sources = source_recipes()
        job = next_job(sources)
        if args.preflight:
            print(json.dumps(job, sort_keys=True, indent=2))
            return 0
        if not job["done"]:
            convert(job, sources, args.stage_timeout)
        return 0
    with MICRO_LOCK.open("a+") as guard:
        try:
            fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("active micro controller; do not start adaptive conversion") from exc
        sources = source_recipes()
        job = next_job(sources)
        if args.preflight:
            print(json.dumps(job, sort_keys=True, indent=2))
            return 0
        if job["done"]:
            return 0
        convert(job, sources, args.stage_timeout)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
