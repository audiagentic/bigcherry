"""Fail-closed controller for one-tensor IQ6R micro-checkpoints.

Distinct from the retired 20-tensor controller. A separate lease is acquired
per stage, so other queued GPU work can run between stages. Never retries a
failed stage, deletes unsealed output, modifies original Stage 1, or promotes
production. Start only once; lock and receipt checks prevent duplicates.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from iq6r_v4_controller import WORK, stage_environment

# The runner imports the NumPy-based calibration preparation module on import.
# The controller must add the same pinned pydeps path as its subprocesses.
sys.path.insert(0, str(WORK / "pydeps"))
from iq6r_v4_micro_convert import (
    MICRO_ROOT,
    MICRO_STATUS,
    micro_recipes,
    predecessor,
    stage_one_record,
)
from iq6r_v4_staged_convert import write_json

STATE = WORK / "conversion-controller-micro.json"
LOCK = WORK / "conversion-controller-micro.lock"


def next_stage(sources: list[bytes]) -> int:
    # Even old receipts with retired artifacts must remain. The newest
    # sealed RAD is checked against its receipt by predecessor().
    receipts = sorted(MICRO_ROOT.glob("micro-*.json")) if MICRO_ROOT.is_dir() else []
    entries: list[int] = []
    for path in receipts:
        # Convert and plan logs are not receipts.
        if len(path.stem) != len("micro-0002") or not path.stem.startswith("micro-"):
            continue
        entries.append(int(path.stem.split("-")[1]))
    if entries != list(range(2, 2 + len(entries))):
        raise RuntimeError("micro stage receipts are not contiguous")
    upcoming = 2 + len(entries)
    if upcoming <= len(sources):
        predecessor(upcoming, sources)
    elif entries:
        # The final stage is complete only if its sealed file/receipt match.
        final = MICRO_ROOT / f"micro-{len(sources):04d}.rad"
        record = json.loads((MICRO_ROOT / f"micro-{len(sources):04d}.json").read_text())
        if (not final.is_file() or record["sha256"] !=
                __import__("iq6r_v4_convert_experimental").sha(final)):
            raise RuntimeError("final micro artifact is absent or unverified")
    return upcoming


def run(stage: int, timeout: int) -> int:
    script = Path(__file__).with_name("iq6r_v4_micro_convert.py")
    command = [
        "agent-lease", "acquire", "gpu", "--wait", "--est", "2h", "--ttl", "4h",
        "--purpose", f"IQ6R-V4 micro checkpoint {stage:04d}",
        "--", sys.executable, "-u", str(script),
        "--stage", str(stage), "--stage-timeout", str(timeout),
    ]
    return subprocess.run(command, env=stage_environment(), check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-timeout", type=int, default=10800)
    parser.add_argument("--max-stages", type=int, default=0,
                        help="optional run limit for manual intervention, 0 means all")
    args = parser.parse_args()
    if not 300 <= args.stage_timeout <= 10800:
        parser.error("stage timeout must be 300..10800")
    if args.max_stages < 0:
        parser.error("max-stages must be >= 0")
    with LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = micro_recipes()
        original = stage_one_record(sources[0])
        next_stage(sources)
        state = {
            "status": "started",
            "stage_total": len(sources),
            "stage_one_sha256": original["sha256"],
            "pid": os.getpid(),
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "production_promotion": False,
            "quality_passed": False,
            "tp3_passed": False,
        }
        write_json(STATE, state)
        processed = 0
        try:
            while True:
                stage = next_stage(sources)
                state["next_stage"] = stage
                state["completed_micro_stages"] = stage - 2
                if stage > len(sources):
                    state["status"] = "fully_sealed_unqualified"
                    write_json(STATE, state)
                    return 0
                if args.max_stages and processed >= args.max_stages:
                    state["status"] = "run_limit_complete_unqualified"
                    write_json(STATE, state)
                    return 0
                state["status"] = "queued_for_exclusive_lease"
                write_json(STATE, state)
                rc = run(stage, args.stage_timeout)
                if rc:
                    raise RuntimeError(f"micro-stage {stage} failed (exit {rc}); no retry")
                current = json.loads(MICRO_STATUS.read_text())
                if current.get("status") != "sealed_unqualified" or current.get("stage") != stage:
                    raise RuntimeError(f"micro-stage {stage} returned success without sealing")
                if next_stage(sources) != stage + 1:
                    raise RuntimeError(f"micro-stage {stage} checkpoint validation failed")
                processed += 1
                state["status"] = "checkpoint_verified"
                state["last_sealed_stage"] = stage
                write_json(STATE, state)
        except BaseException as exc:
            state["status"] = "blocked_no_retry"
            state["error"] = str(exc)
            write_json(STATE, state)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
