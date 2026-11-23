"""Release-locked IQ6R-V4 conversion. Run only within agent-lease acquire gpu.

Production model paths are never touched. Every IQ6R GPTQ input must have an
expanded RADGRAM2 Hessian; missing-calibration fallback invalidates an output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from iq6r_v4_fetch_hessians import manifest_entries
from iq6r_v4_offload import quantizer_offload_arches
from iq6r_v4_prepare_grams import HEADER, MAGIC, ROWS, check_and_map
from iq6r_v4_validate_hessians import audit

ROOT = Path(os.environ["BC_WORK_ROOT"]) / "iq6r-v4-qualification"
ORIGINAL = Path(os.environ["BC_MODEL_ROOT"]) / "qwen3.8-27b/Qwen3.8-27B"
STAGED = ROOT / "vhead-5.0"
RAW_HESSIANS = ROOT / "upstream-hessians"
HESSIANS = ROOT / "calib-radgram2"
PINNED = Path(os.environ["BC_RADIANCE_SOURCE"])
RELEASE = Path(os.environ["BC_RELEASE_MANIFEST"])
VHEAD_SHA256 = "1dd39da8c0a5205fd1f53e2052807bafe0a1c2fb291a0a88908df427423e5f90"
RECIPE_SHA256 = "0c41614302d448ba61c39f31eb6da1ee4970fcd4fa1a6011ec479d51b64e597f"
ARTIFACT = ROOT / "qwen3.8-27b-iq6r-v4-experimental.rad"
TEMP = ROOT / "qwen3.8-27b-iq6r-v4-unverified.rad"
RECEIPT = ROOT / "conversion-status-managed.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(data: dict) -> None:
    part = RECEIPT.with_suffix(".tmp")
    part.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    part.replace(RECEIPT)


def managed_build() -> tuple[Path, dict]:
    manifest = json.loads(RELEASE.read_text())
    if manifest.get("tag") != "bc-radiance-1170.26.0" or manifest.get("source") != "ci":
        raise ValueError("unexpected managed release identity")
    build = Path(manifest["build_dir"])
    if sha(build / "bin/radiance") != manifest["binary_sha256"]:
        raise ValueError("managed runtime binary differs from release manifest")
    for relative in ("bin/rad-convert", "radiance_home/kernels/libr3.so",
                     "radiance_home/kernels/libr4d.so"):
        if not (build / relative).is_file():
            raise ValueError(f"managed release missing {relative}")
    return build, manifest


def verified_inputs(build: Path) -> dict:
    receipt = json.loads((ROOT / "source-sha256-recheck.json").read_text())
    if not receipt.get("all_match") or receipt.get("count") != 18:
        raise ValueError("original BF16 checkpoint fingerprints were not verified")
    if sha(STAGED / "vhead.safetensors") != VHEAD_SHA256:
        raise ValueError("vocabulary head changed since staging")
    recipe = PINNED / "data/recipes/q38-27b-iq6r-v4.recipe"
    if sha(recipe) != RECIPE_SHA256:
        raise ValueError("pinned recipe differs from recorded source")
    entries = manifest_entries(json.loads((ROOT / "hessian-tree.json").read_text()))
    for entry in entries:
        file = RAW_HESSIANS / Path(entry["path"]).name
        if not file.is_file() or file.stat().st_size != entry["bytes"]:
            raise ValueError(f"missing/incomplete upstream Hessian: {file.name}")
        if sha(file) != entry["sha256"]:
            raise ValueError(f"upstream Hessian SHA-256 mismatch: {file.name}")
    audit(RAW_HESSIANS, ORIGINAL, ROWS)
    mapping = check_and_map(entries, recipe)
    progress = json.loads((HESSIANS / "adapter-progress.json").read_text())
    if progress.get("completed") != 257 or progress.get("total") != 257:
        raise ValueError("RADGRAM2 calibration preparation is incomplete")
    sizes = {}
    for entry in entries:
        src = RAW_HESSIANS / Path(entry["path"]).name
        with src.open("rb") as stream:
            n = int.from_bytes(stream.read(8), "little")
            meta = json.loads(stream.read(n))["__metadata__"]
        sizes[entry["path"]] = int(meta["k"])
    for stem, entry in mapping.items():
        k = sizes[entry["path"]]
        file = HESSIANS / f"gram.r0.{stem}.bin"
        if not file.is_file() or file.stat().st_size != HEADER.size + 4*k*k:
            raise ValueError(f"missing or truncated RADGRAM2: {stem}")
        with file.open("rb") as stream:
            if stream.read(HEADER.size) != HEADER.pack(MAGIC, k, 0, ROWS, 0, 0):
                raise ValueError(f"invalid RADGRAM2 header: {stem}")
    return {
        "source": "18/18 BF16 shard SHA256 matches prior reference",
        "vhead_sha256": VHEAD_SHA256,
        "recipe_sha256": RECIPE_SHA256,
        "dataset_revision": "2dcf0061cb683152742b904e2965ce7cd92b2da6",
        "source_hessians_sha256_verified": len(entries),
        "radgram2_calibrated_stems": len(mapping),
        "converter_sha256": sha(build / "bin/rad-convert"),
        "calibration_rows": ROWS,
    }


def arguments(build: Path, kernels: str, output: str, *, plan: bool) -> list[str]:
    cmd = [
        str(build / "bin/rad-convert"), str(STAGED), "-o", output,
        "--home", str(build / "radiance_home"), "--kernels", kernels,
        "--recipe", str(PINNED / "data/recipes/q38-27b-iq6r-v4.recipe"),
        "--set", "quantization_config.quant_method=w8a8",
        "--set", "quantization_config.fmt=fwht128",
        "--report", "25", "-v",
    ]
    if plan:
        cmd.append("--plan-only")
    else:
        cmd += ["--gpus", "0"]
    return cmd


def run_process(args: list[str], env: dict, label: str, limit: int) -> int:
    stderr_path = ROOT / f"managed-{label}-stderr.log"
    with (ROOT / f"managed-{label}-stdout.log").open("w") as out, stderr_path.open("w") as err:
        try:
            child = subprocess.run(args, env=env, stdout=out, stderr=err,
                                   timeout=limit, check=False)
            result = child.returncode
        except subprocess.TimeoutExpired:
            result = 124
    if "no calibration for" in stderr_path.read_text(errors="replace"):
        raise RuntimeError(f"missing GPTQ calibration fallback during {label}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert", action="store_true", help="actually emit experimental RAD")
    args = parser.parse_args()
    lease = subprocess.run(["agent-lease", "status"], capture_output=True,
                           text=True, check=True).stdout
    if "gpu: held" not in lease or "IQ6R" not in lease:
        raise RuntimeError("run only under an IQ6R exclusive GPU lease")
    if TEMP.exists() or ARTIFACT.exists():
        raise RuntimeError("refusing overwrite of existing RAD artifact")
    build, release = managed_build()
    status = {
        "stage": "validating_source_and_calibration",
        "production_promotion": False,
        "release_tag": release["tag"],
        "build_dir": str(build),
        "release_manifest": release,
    }
    status["quantizer_offload_arches"] = quantizer_offload_arches(build / "radiance_home/quantizers/libquant.so", "gfx1201")
    started = time.monotonic()
    save(status)
    try:
        status["verified_inputs"] = verified_inputs(build)
        status["stage"] = "gfx1100_plan"
        save(status)
        env = os.environ.copy()
        env["CALIB"] = str(HESSIANS)
        env["HIP_VISIBLE_DEVICES"] = "0,1"
        cmd = arguments(build, "libr3,libref", "/dev/null", plan=True)
        status["gfx1100_plan"] = cmd
        save(status)
        code = run_process(cmd, env, "plan-gfx1100", 180)
        if code:
            raise RuntimeError(f"managed gfx1100 plan failed ({code})")
        status["stage"] = "gfx1201_plan"
        save(status)
        env["HIP_VISIBLE_DEVICES"] = "2"
        cmd = arguments(build, "libr4d,libref", "/dev/null", plan=True)
        status["gfx1201_plan"] = cmd
        save(status)
        code = run_process(cmd, env, "plan-gfx1201", 180)
        if code:
            raise RuntimeError(f"managed gfx1201 plan failed ({code})")
        if not args.convert:
            status["stage"] = "managed_release_plan_only_passed"
            save(status)
            return 0
        status["stage"] = "converting_experimental"
        save(status)
        env["HIP_VISIBLE_DEVICES"] = "2"
        cmd = arguments(build, "libr4d,libref", str(TEMP), plan=False)
        status["conversion_cmd"] = cmd
        save(status)
        code = run_process(cmd, env, "conversion", 14400)
        if code:
            raise RuntimeError(f"managed converter exited with {code}")
        if not TEMP.is_file() or TEMP.stat().st_size < 5 * 1024**3:
            raise RuntimeError("RAD output missing or unexpectedly small")
        status["stage"] = "hashing_experimental_model"
        save(status)
        status["artifact_sha256"] = sha(TEMP)
        status["artifact_bytes"] = TEMP.stat().st_size
        TEMP.rename(ARTIFACT)
        status["stage"] = "experimental_RAD_ready_unqualified"
        status["artifact"] = str(ARTIFACT)
        return 0
    except Exception as exc:
        status["stage"] = "failed_no_production_change"
        status["error"] = str(exc)
        raise
    finally:
        status["elapsed_s"] = round(time.monotonic() - started, 2)
        save(status)


if __name__ == "__main__":
    raise SystemExit(main())
