"""Detached, fail-closed controller for IQ6R-V4 staged conversion.

Run ONCE without a GPU lease while the independently leased Stage 1 is active.
After Stage 1 has exited and sealed its RAD2, run stages 2..16 under separate
exclusive GPU leases. Incomplete, failed or incompatible checkpoints stop the
controller; never retry, overwrite, promote, or run production/TP3 jobs.
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

from iq6r_v4_audit_artifact import parse_radinfo, recipe_codes
from iq6r_v4_stage_recipes import PINNED_RECIPE_SHA256

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORK = Path(os.environ["BC_WORK_ROOT"]) / "iq6r-v4-qualification"
STAGED = WORK / "staged-conversion"
STAGES_STATUS = WORK / "conversion-status-staged.json"
CONTROLLER_STATUS = WORK / "conversion-controller.json"
LOCK = WORK / "conversion-controller.lock"
FINAL = WORK / "qwen3.8-27b-iq6r-v4-experimental.rad"
SOURCE = Path(os.environ["BC_RADIANCE_SOURCE"])
RECIPE = SOURCE / "data/recipes/q38-27b-iq6r-v4.recipe"
BUILD_MANIFEST = Path(os.environ["BC_RELEASE_MANIFEST"])
STAGE_COUNT = 16
CHUNK = 20


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def check_stages(data: dict) -> int:
    """Validate contiguous sealed receipts and the latest retained checkpoint."""
    if data.get("stages") != STAGE_COUNT or data.get("chunk") != CHUNK:
        raise RuntimeError("staged manifest dimensions have changed")
    if data.get("full_recipe_sha256") != PINNED_RECIPE_SHA256:
        raise RuntimeError("staged recipe identity changed")
    if data.get("release") != "bc-radiance-1170.26.0":
        raise RuntimeError("managed release differs from experiment")
    if data.get("production_promotion") is not False:
        raise RuntimeError("unexpected production flag")
    done = data.get("done")
    if not isinstance(done, list) or len(done) > STAGE_COUNT:
        raise RuntimeError("malformed staged receipts")
    for i, entry in enumerate(done, 1):
        receipt = STAGED / f"stage-{i:02d}.json"
        if entry.get("stage") != i or not receipt.is_file():
            raise RuntimeError(f"stage {i} missing or unordered receipt")
        if json.loads(receipt.read_text()) != entry:
            raise RuntimeError(f"stage {i} receipt differs from controller record")
        recipe_path = STAGED / f"stage-{i:02d}.recipe"
        if not recipe_path.is_file() or sha(recipe_path) != entry.get("recipe_sha256"):
            raise RuntimeError(f"stage {i} recipe hash mismatch")
    if done:
        latest = done[-1]
        artifact = STAGED / f"stage-{len(done):02d}.rad"
        if len(done) == STAGE_COUNT and FINAL.is_file() and not artifact.exists():
            artifact = FINAL
        if (not artifact.is_file() or artifact.stat().st_size != latest["bytes"]
                or sha(artifact) != latest["sha256"]):
            raise RuntimeError("latest checkpoint is missing or modified")
    return len(done)


def validate_reuse(stage: int) -> dict:
    """Ensure previous GPTQ weights were copied, not recomputed on the GPU."""
    if stage < 2:
        return {"checked": False}
    path = STAGED / f"stage-{stage:02d}-convert-stderr.log"
    if not path.is_file():
        raise RuntimeError(f"stage {stage} conversion log missing")
    log = path.read_text(errors="replace")
    # All earlier explicit IQ6R tensors must be eligible for reuse. At stage
    # two the shared vocabulary head is the critical initial proof.
    if re.search(r"GPTQ quantises 'output\.k[4-7](?:\.weight)?'", log):
        raise RuntimeError("vocabulary-head GPTQ unexpectedly recomputed")
    matches = re.findall(r"--reuse: (\d+) weight\(s\)", log)
    if not matches or int(matches[-1]) < 4:
        raise RuntimeError(f"stage {stage} lacks evidence of reused weights")
    if "no calibration for" in log:
        raise RuntimeError("missing-calibration fallback detected")
    return {"checked": True, "copied_weights": int(matches[-1]),
            "head_requantized": False}


def finish(data: dict, state: dict) -> None:
    if check_stages(data) != STAGE_COUNT:
        raise RuntimeError("cannot publish with incomplete staged checkpoints")
    expected = RECIPE.read_bytes()
    if hashlib.sha256(expected).hexdigest() != PINNED_RECIPE_SHA256:
        raise RuntimeError("source recipe changed")
    full_recipe_path = STAGED / "stage-16.recipe"
    if full_recipe_path.read_bytes() != expected:
        raise RuntimeError("final stage is not the exact pinned recipe")
    last = data["done"][-1]
    if FINAL.exists():
        if sha(FINAL) != last["sha256"] or FINAL.stat().st_size != last["bytes"]:
            raise RuntimeError("existing experimental final artifact differs")
        artifact = FINAL
    else:
        artifact = STAGED / "stage-16.rad"
        if not artifact.is_file():
            raise RuntimeError("final sealed RAD2 missing")
    report = STAGED / "stage-16-radinfo.txt"
    if not report.is_file():
        raise RuntimeError("final rad-info record missing")
    info = parse_radinfo(report.read_text())
    wanted = recipe_codes(expected)
    if info["trellis_codes"] != dict(wanted):
        raise RuntimeError(f"final trellis counts differ: {info['trellis_codes']}")
    if not FINAL.exists():
        artifact.replace(FINAL)
    state["state"] = "full_recipe_artifact_ready_unqualified"
    state["artifact"] = str(FINAL)
    state["artifact_bytes"] = last["bytes"]
    state["artifact_sha256"] = last["sha256"]
    state["numerical_quality_passed"] = False
    state["tp3_passed"] = False
    state["production_promotion"] = False
    atomic_json(CONTROLLER_STATUS, state)


def stage_environment() -> dict[str, str]:
    env = os.environ.copy()
    deps = WORK / "pydeps"
    if not (deps / "numpy").is_dir():
        raise RuntimeError("experiment NumPy dependency directory is missing")
    env["PYTHONPATH"] = str(deps) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def lease_command(stage: int, stage_timeout: int) -> list[str]:
    script = Path(__file__).with_name("iq6r_v4_staged_convert.py")
    return [
        "agent-lease", "acquire", "gpu", "--wait", "--est", "2h",
        "--ttl", "4h", "--purpose", f"IQ6R staged R9700 stage {stage:02d}",
        "--", sys.executable, "-u", str(script),
        "--chunk", str(CHUNK), "--stage-limit", str(stage),
        "--stage-timeout", str(stage_timeout),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stage-timeout", type=int, default=10800)
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 300:
        parser.error("poll-seconds must be between 5 and 300")
    if not 300 <= args.stage_timeout <= 10800:
        parser.error("stage-timeout must be 300-10800 seconds")
    WORK.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another IQ6R controller is already running") from exc
        if not STAGES_STATUS.is_file():
            raise RuntimeError("Stage 1 was never started")
        if sha(RECIPE) != PINNED_RECIPE_SHA256:
            raise RuntimeError("pinned IQ6R recipe has drifted")
        manifest = json.loads(BUILD_MANIFEST.read_text())
        if manifest["tag"] != "bc-radiance-1170.26.0":
            raise RuntimeError("release manifest tag changed")
        state = {
            "state": "waiting_for_stage_1", "pid": os.getpid(),
            "stage_count": STAGE_COUNT, "release": manifest["tag"],
            "production_promotion": False, "quality_passed": False,
            "tp3_passed": False,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(CONTROLLER_STATUS, state)
        first_stage_free_polls = 0

        try:
            while True:
                data = json.loads(STAGES_STATUS.read_text())
                completed = check_stages(data)
                state["completed_stages"] = completed
                if completed == STAGE_COUNT:
                    finish(data, state)
                    return 0
                next_stage = completed + 1
                if next_stage == 1:
                    # Stage 1 was launched separately and owns the GPU lease.
                    # Never launch a second stage-1 worker.
                    lease_status = subprocess.check_output(
                        ["agent-lease", "status"], text=True)
                    if "gpu: held" in lease_status:
                        first_stage_free_polls = 0
                        state["state"] = "waiting_for_existing_stage_1"
                        atomic_json(CONTROLLER_STATUS, state)
                        time.sleep(args.poll_seconds)
                        continue
                    first_stage_free_polls += 1
                    if first_stage_free_polls >= 2:
                        raise RuntimeError("Stage 1 stopped without a sealed checkpoint; inspect its failure log")
                    time.sleep(args.poll_seconds)
                    continue

                state["state"] = "starting_next_stage"
                state["next_stage"] = next_stage
                atomic_json(CONTROLLER_STATUS, state)
                command = lease_command(next_stage, args.stage_timeout)
                # Agent-lease --wait queues behind any legitimate other owner.
                # Do not retry a failed stage; leave the diagnostic output.
                result = subprocess.run(command, check=False, env=stage_environment())
                if result.returncode:
                    raise RuntimeError(
                        f"stage {next_stage} failed under lease (exit {result.returncode}); "
                        f"inspect staged-conversion logs")
                data = json.loads(STAGES_STATUS.read_text())
                if check_stages(data) < next_stage:
                    raise RuntimeError(f"stage {next_stage} exited without sealing its checkpoint")
                state.setdefault("reuse_evidence", {})[str(next_stage)] = validate_reuse(next_stage)
                state["state"] = "checkpoint_verified"
                state["completed_stages"] = next_stage
                atomic_json(CONTROLLER_STATUS, state)
        except Exception as exc:
            state["state"] = "blocked_no_retry"
            state["error"] = str(exc)
            state["production_promotion"] = False
            atomic_json(CONTROLLER_STATUS, state)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
