"""Single-GPTQ-tensor, independently sealed IQ6R-V4 conversion.

Stage 1 is the existing verified four-head checkpoint from the original 20-
tensor schedule. Micro-stage 2 adds *one* trunk GPTQ weight and copies all
other tensors from Stage 1; subsequent stages do likewise from the previous
micro checkpoint. The failed original stage-02 file is never read or modified.

This program performs ONE explicitly selected stage per invocation. GPU runs
require an external exclusive IQ6R agent-lease; --preflight is card-free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from iq6r_v4_convert_experimental import HESSIANS, PINNED, ROOT, managed_build, sha
from iq6r_v4_offload import quantizer_offload_arches
from iq6r_v4_stage_recipes import PINNED_RECIPE_SHA256, recipes
from iq6r_v4_staged_convert import (
    STAGES_ROOT,
    STATUS,
    call,
    desired_encoding_counts,
    runner_command,
    verify_stage_info,
    write_json,
)

MICRO_ROOT = ROOT / "micro-conversion"
MICRO_STATUS = ROOT / "conversion-status-micro.json"
MICRO_CHUNK = 1
ORIGINAL_STAGE_ONE = STAGES_ROOT / "stage-01.rad"
ORIGINAL_STAGE_ONE_RECIPE = STAGES_ROOT / "stage-01.recipe"


def micro_recipes() -> list[bytes]:
    source = PINNED / "data/recipes/q38-27b-iq6r-v4.recipe"
    fallback = PINNED / "data/recipes/q38-27b-w8a8-df2.recipe"
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != PINNED_RECIPE_SHA256:
        raise ValueError("primary recipe SHA drifted")
    return recipes(data, fallback.read_bytes(), chunk=MICRO_CHUNK)


def stage_one_record(generated: bytes) -> dict:
    """Fail closed on *original* immutable sealed Stage 1 identity."""
    old = json.loads(STATUS.read_text())
    if old.get("release") != "bc-radiance-1170.26.0" or old.get("chunk") != 20:
        raise RuntimeError("original Stage 1 identity differs")
    done = old.get("done", [])
    if not done or done[0].get("stage") != 1:
        raise RuntimeError("original Stage 1 was not sealed")
    record = done[0]
    if not ORIGINAL_STAGE_ONE_RECIPE.is_file() or ORIGINAL_STAGE_ONE_RECIPE.read_bytes() != generated:
        raise RuntimeError("original Stage 1 recipe differs from micro first stage")
    if (not ORIGINAL_STAGE_ONE.is_file()
            or ORIGINAL_STAGE_ONE.stat().st_size != record["bytes"]
            or sha(ORIGINAL_STAGE_ONE) != record["sha256"]
            or hashlib.sha256(generated).hexdigest() != record["recipe_sha256"]):
        raise RuntimeError("original Stage 1 hash or recipe differs")
    return record


def predecessor(stage: int, stages: list[bytes]) -> tuple[Path, dict]:
    if stage == 2:
        return ORIGINAL_STAGE_ONE, stage_one_record(stages[0])
    previous = stage - 1
    path = MICRO_ROOT / f"micro-{previous:04d}.rad"
    receipt = MICRO_ROOT / f"micro-{previous:04d}.json"
    if not path.is_file() or not receipt.is_file():
        raise RuntimeError(f"micro-stage {previous} is not sealed")
    record = json.loads(receipt.read_text())
    if (record.get("stage") != previous or
            record.get("recipe_sha256") != hashlib.sha256(stages[previous - 1]).hexdigest() or
            record.get("sha256") != sha(path) or record.get("bytes") != path.stat().st_size):
        raise RuntimeError(f"micro-stage {previous} receipt or artifact differs")
    return path, record


def planned_stage(stage: int, sources: list[bytes], stage_one: dict) -> dict:
    if not 2 <= stage <= len(sources):
        raise ValueError("micro stage outside 2..289")
    return {
        "status": "preflight_only",
        "stage": stage,
        "total_stages": len(sources),
        "gptq_before": 4 + stage - 2,
        "gptq_after": 4 + stage - 1,
        "recipe_sha256": hashlib.sha256(sources[stage - 1]).hexdigest(),
        "original_stage_one_sha256": stage_one["sha256"],
        "production_promotion": False,
        "model_qualified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--preflight", action="store_true",
                        help="validate source and Stage 1 only; no GPU or outputs")
    parser.add_argument("--stage-timeout", type=int, default=10800)
    args = parser.parse_args()
    if not 300 <= args.stage_timeout <= 10800:
        parser.error("stage timeout must be in [300, 10800]")
    sources = micro_recipes()
    if not 2 <= args.stage <= len(sources):
        parser.error("stage must be in 2..289")
    # Every invocation must ensure the original Stage 1 was sealed, not just
    # that some later predecessor exists.
    original = stage_one_record(sources[0])
    plan = planned_stage(args.stage, sources, original)
    if args.preflight:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    lease = __import__("subprocess").run(
        ["agent-lease", "status"], capture_output=True, text=True, check=True).stdout
    if "gpu: held" not in lease or "IQ6R" not in lease:
        raise RuntimeError("exclusive IQ6R GPU lease required")
    build, manifest = managed_build()
    old = json.loads(STATUS.read_text())
    if manifest["tag"] != old["release"] or manifest["binary_sha256"] != old["binary_sha256"]:
        raise RuntimeError("managed build changed since sealed Stage 1")
    if sha(build / "bin/rad-convert") != old["converter_sha256"]:
        raise RuntimeError("converter identity changed since sealed Stage 1")
    quantizer_offload_arches(build / "radiance_home/quantizers/libquant.so", "gfx1201")

    MICRO_ROOT.mkdir(parents=True, exist_ok=True)
    # Validate last predecessor before any output can be opened.
    prior, prior_record = predecessor(args.stage, sources)
    name = f"micro-{args.stage:04d}"
    output = MICRO_ROOT / f"{name}.rad"
    incomplete = MICRO_ROOT / f"{name}.incomplete.rad"
    receipt = MICRO_ROOT / f"{name}.json"
    stage_recipe = MICRO_ROOT / f"{name}.recipe"
    if output.exists() or incomplete.exists() or receipt.exists():
        raise RuntimeError(f"refusing to overwrite existing micro-stage {args.stage} output")
    recipe_bytes = sources[args.stage - 1]
    if stage_recipe.exists() and stage_recipe.read_bytes() != recipe_bytes:
        raise RuntimeError("micro-stage recipe was modified")
    if not stage_recipe.exists():
        stage_recipe.write_bytes(recipe_bytes)

    env = os.environ.copy()
    env["CALIB"] = str(HESSIANS)
    env["HIP_VISIBLE_DEVICES"] = "2"
    env.pop("ROCR_VISIBLE_DEVICES", None)
    run_status = {
        **plan, "status": "running_stage", "release": manifest["tag"],
        "predecessor_sha256": prior_record["sha256"],
        "incomplete_output": str(incomplete),
    }
    write_json(MICRO_STATUS, run_status)
    plan_cmd = runner_command(build, stage_recipe, Path("/dev/null"), prior, plan_only=True)
    if call(plan_cmd, env, MICRO_ROOT / f"{name}-plan-stdout.log",
            MICRO_ROOT / f"{name}-plan-stderr.log", 180) != 0:
        raise RuntimeError("micro-stage preflight plan-only failed")
    convert_cmd = runner_command(build, stage_recipe, incomplete, prior, plan_only=False)
    start = __import__("time").monotonic()
    rc = call(convert_cmd, env, MICRO_ROOT / f"{name}-stdout.log",
              MICRO_ROOT / f"{name}-stderr.log", args.stage_timeout)
    if rc:
        run_status["status"] = "blocked_no_retry"
        run_status["exit_code"] = rc
        write_json(MICRO_STATUS, run_status)
        raise RuntimeError(f"micro-stage {args.stage} conversion failed with {rc}")
    if not incomplete.is_file() or incomplete.stat().st_size < 5 * 1024**3:
        raise RuntimeError("micro-stage output truncated")
    info_proc = __import__("subprocess").run(
        [str(build / "bin/rad-info"), str(incomplete)],
        capture_output=True, text=True, timeout=300, check=False)
    (MICRO_ROOT / f"{name}-radinfo.txt").write_text(info_proc.stdout + info_proc.stderr)
    if info_proc.returncode:
        raise RuntimeError("micro-stage rad-info failed")
    info = verify_stage_info(info_proc.stdout, recipe_bytes)
    # Ensure exactly one new GPTQ rule, not just a valid container.
    actual = sum(desired_encoding_counts(recipe_bytes).values())
    if actual != 4 + args.stage - 1:
        raise RuntimeError("unexpected GPTQ rule count")
    record = {
        "stage": args.stage, "total_stages": len(sources),
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "sha256": sha(incomplete), "bytes": incomplete.stat().st_size,
        "gptq_cumulative": actual, "radinfo": info,
        "predecessor_sha256": prior_record["sha256"],
        "elapsed_s": round(__import__("time").monotonic() - start, 2),
        "production_promotion": False, "complete_model_qualified": False,
    }
    incomplete.rename(output)
    write_json(receipt, record)
    run_status.update({
        "status": "sealed_unqualified", "sha256": record["sha256"],
        "gptq_cumulative": actual,
    })
    write_json(MICRO_STATUS, run_status)
    # Keep one previous micro checkpoint and the latest; the failed old
    # stage-02 and independently sealed original Stage 1 are never modified.
    if args.stage >= 4:
        (MICRO_ROOT / f"micro-{args.stage - 2:04d}.rad").unlink(missing_ok=True)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
