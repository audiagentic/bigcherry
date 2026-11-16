"""Resumable, stage-sealed IQ6R-V4 full-model conversion on managed gfx1201.

Run under an exclusive agent-lease gpu lock. Each stage writes a COMPLETE RAD2
with a cumulative GPTQ recipe; the next stage reuses matching converted tensors.
Only the final recipe is the fully qualified V4 encoding. No production writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

from iq6r_v4_audit_artifact import parse_radinfo
from iq6r_v4_convert_experimental import (
    HESSIANS,
    PINNED,
    ROOT,
    STAGED,
    managed_build,
    sha,
    verified_inputs,
)
from iq6r_v4_offload import quantizer_offload_arches
from iq6r_v4_stage_recipes import PINNED_RECIPE_SHA256, recipes

STAGES_ROOT = ROOT / "staged-conversion"
STATUS = ROOT / "conversion-status-staged.json"
FINAL = ROOT / "qwen3.8-27b-iq6r-v4-experimental.rad"


def write_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def desired_encoding_counts(recipe_bytes: bytes) -> Counter[str]:
    counters: Counter[str] = Counter()
    for line in recipe_bytes.decode().splitlines():
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if tokens[1] != "gptq":
            continue
        codes = [p.split("=", 1)[1] for p in tokens if p.startswith("codes=")]
        if len(codes) != 1:
            raise ValueError(f"unexpected GPTQ rule in staged recipe: {line}")
        counters[codes[0]] += 1
    return counters


def verify_stage_info(text: str, source: bytes) -> dict:
    info = parse_radinfo(text)
    desired = desired_encoding_counts(source)
    if Counter(info["trellis_codes"]) != desired:
        raise RuntimeError(
            f"RAD2 trellis count mismatch: actual={info['trellis_codes']}, expected={dict(desired)}"
        )
    return info


def runner_command(build: Path, recipe: Path, output: Path, reuse: Path | None,
                   *, plan_only: bool) -> list[str]:
    result = [
        str(build / "bin/rad-convert"), str(STAGED), "-o", str(output),
        "--home", str(build / "radiance_home"), "--kernels", "libr4d,libref",
        "--recipe", str(recipe),
        "--set", "quantization_config.quant_method=w8a8",
        "--set", "quantization_config.fmt=fwht128",
        "--report", "25", "-v",
    ]
    if reuse is not None:
        result.extend(["--reuse", str(reuse)])
    if plan_only:
        result.append("--plan-only")
    else:
        result.extend(["--gpus", "0"])
    return result


def call(cmd: list[str], env: dict, stdout: Path, stderr: Path, timeout: int) -> int:
    with stdout.open("w") as out, stderr.open("w") as err:
        try:
            proc = subprocess.run(cmd, env=env, stdout=out, stderr=err,
                                  timeout=timeout, check=False)
            ret = proc.returncode
        except subprocess.TimeoutExpired:
            ret = 124
    if "no calibration for" in stderr.read_text(errors="replace"):
        raise RuntimeError(f"missing calibration fallback in {stderr}")
    return ret


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--stage-limit", type=int, default=0,
                        help="0 for all stages, otherwise stop after this cumulative stage")
    parser.add_argument("--stage-timeout", type=int, default=10800)
    args = parser.parse_args()

    lease = subprocess.run(["agent-lease", "status"], text=True, capture_output=True,
                           check=True).stdout
    if "gpu: held" not in lease or "IQ6R" not in lease:
        raise RuntimeError("exclusive IQ6R GPU lease is mandatory")
    if FINAL.exists():
        raise FileExistsError(f"refusing to replace final experimental RAD: {FINAL}")
    if args.stage_timeout < 300 or args.stage_timeout > 43200:
        raise ValueError("stage timeout must be between 300 and 43200 seconds")

    recipe = PINNED / "data/recipes/q38-27b-iq6r-v4.recipe"
    content = recipe.read_bytes()
    if hashlib.sha256(content).hexdigest() != PINNED_RECIPE_SHA256:
        raise ValueError("pinned recipe SHA mismatch")
    baseline = (PINNED / "data/recipes/q38-27b-w8a8-df2.recipe").read_bytes()
    sources = recipes(content, baseline, chunk=args.chunk)
    STAGES_ROOT.mkdir(parents=True, exist_ok=True)
    build, manifest = managed_build()
    arches = quantizer_offload_arches(
        build / "radiance_home/quantizers/libquant.so", "gfx1201")
    env = os.environ.copy()
    env["CALIB"] = str(HESSIANS)
    env["HIP_VISIBLE_DEVICES"] = "2"
    env.pop("ROCR_VISIBLE_DEVICES", None)
    if args.stage_limit and not 1 <= args.stage_limit <= len(sources):
        raise ValueError("stage limit outside staged recipe range")
    status = {
        "status": "started",
        "stages": len(sources),
        "chunk": args.chunk,
        "release": manifest["tag"],
        "binary_sha256": manifest["binary_sha256"],
        "converter_sha256": sha(build / "bin/rad-convert"),
        "quantizer_offload_arches": arches,
        "full_recipe_sha256": PINNED_RECIPE_SHA256,
        "done": [],
        "production_promotion": False,
        "quality_passed": False,
        "tp3_passed": False,
    }

    if STATUS.exists():
        previous_status = json.loads(STATUS.read_text())
        for k in ("chunk", "stages", "release", "binary_sha256",
                  "converter_sha256", "full_recipe_sha256"):
            if previous_status.get(k) != status[k]:
                raise ValueError(f"incompatible existing staged checkpoint metadata: {k}")
        status["done"] = previous_status.get("done", [])

    # Validate hashes against independent source receipts once before first stage.
    if not status["done"]:
        status["verified_inputs"] = verified_inputs(build)
    elif "verified_inputs" in previous_status:
        status["verified_inputs"] = previous_status["verified_inputs"]
    else:
        raise ValueError("prior checkpoint lacks verified input identity")

    prior: Path | None = None
    for i, source in enumerate(sources, 1):
        stage_recipe = STAGES_ROOT / f"stage-{i:02d}.recipe"
        if stage_recipe.exists() and stage_recipe.read_bytes() != source:
            raise ValueError(f"modified existing stage recipe: {stage_recipe}")
        if not stage_recipe.exists():
            stage_recipe.write_bytes(source)

        complete = STAGES_ROOT / f"stage-{i:02d}.rad"
        pending = STAGES_ROOT / f"stage-{i:02d}.incomplete.rad"
        stage_receipt = STAGES_ROOT / f"stage-{i:02d}.json"

        if i <= len(status["done"]):
            record = status["done"][i - 1]
            if record.get("stage") != i or record.get("recipe_sha256") != hashlib.sha256(source).hexdigest():
                raise RuntimeError(f"invalid prior stage checkpoint identity: {i}")
            if i == len(status["done"]) and (not complete.is_file() or complete.stat().st_size != record["bytes"] or sha(complete) != record["sha256"]):
                raise RuntimeError(f"previously completed stage is missing/changed: {i}")
            if i == len(status["done"]):
                prior = complete
            continue

        if args.stage_limit and i > args.stage_limit:
            status["status"] = "stage_limit_complete_unqualified"
            write_json(STATUS, status)
            return 0
        if pending.exists() or complete.exists():
            raise RuntimeError(f"orphaned stage output; refusing overwrite: {pending}, {complete}")
        if i > 1 and (prior is None or not prior.is_file()):
            raise RuntimeError(f"missing sealed input checkpoint for stage {i}")
        status["status"] = "running_stage"
        status["current_stage"] = i
        status["current_recipe_sha256"] = hashlib.sha256(source).hexdigest()
        status["current_output"] = str(pending)
        write_json(STATUS, status)

        plan_cmd = runner_command(build, stage_recipe, Path("/dev/null"), prior,
                                  plan_only=True)
        plan_err = STAGES_ROOT / f"stage-{i:02d}-plan-stderr.log"
        if call(plan_cmd, env, STAGES_ROOT / f"stage-{i:02d}-plan-stdout.log",
                plan_err, 180) != 0:
            raise RuntimeError(f"stage {i} plan-only failed: {plan_err}")

        cmd = runner_command(build, stage_recipe, pending, prior, plan_only=False)
        status["current_command"] = cmd
        write_json(STATUS, status)
        start = time.monotonic()
        stderr_path = STAGES_ROOT / f"stage-{i:02d}-convert-stderr.log"
        code = call(cmd, env, STAGES_ROOT / f"stage-{i:02d}-convert-stdout.log",
                    stderr_path, args.stage_timeout)
        if code != 0:
            raise RuntimeError(f"stage {i} conversion exited {code}; see {stderr_path}")
        if not pending.is_file() or pending.stat().st_size < 5 * 1024 ** 3:
            raise RuntimeError(f"stage {i} output missing/truncated")

        info_file = STAGES_ROOT / f"stage-{i:02d}-radinfo.txt"
        info_run = subprocess.run(
            [str(build / "bin/rad-info"), str(pending)], capture_output=True,
            text=True, timeout=300, check=False)
        info_file.write_text(info_run.stdout + info_run.stderr)
        if info_run.returncode:
            raise RuntimeError(f"stage {i} rad-info failed ({info_run.returncode})")
        info = verify_stage_info(info_run.stdout, source)
        fingerprint = sha(pending)
        size = pending.stat().st_size
        pending.rename(complete)
        record = {
            "stage": i,
            "recipe_sha256": hashlib.sha256(source).hexdigest(),
            "sha256": fingerprint,
            "bytes": size,
            "gptq_cumulative": sum(desired_encoding_counts(source).values()),
            "radinfo": info,
            "elapsed_s": round(time.monotonic() - start, 2),
            "complete_model_qualified": False,
        }
        write_json(stage_receipt, record)
        status["done"].append(record)
        status["status"] = "stage_complete_unqualified"
        write_json(STATUS, status)
        # After a newer complete/hash-checked checkpoint, retire older copies.
        if i >= 3:
            (STAGES_ROOT / f"stage-{i-2:02d}.rad").unlink(missing_ok=True)
        prior = complete

        if args.stage_limit and i >= args.stage_limit:
            status["status"] = "stage_limit_complete_unqualified"
            write_json(STATUS, status)
            return 0

    if len(status["done"]) != len(sources):
        raise RuntimeError("not all IQ6R conversion stages were sealed")
    if prior is None:
        raise RuntimeError("final stage output missing")
    # A complete staged artifact remains explicitly unqualified until the
    # immutable final artifact audit, quality and TP3 tests pass.
    final_sha = sha(prior)
    status["final_sha256"] = final_sha
    status["final_bytes"] = prior.stat().st_size
    status["status"] = "full_recipe_sealed_unqualified"
    write_json(STATUS, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
