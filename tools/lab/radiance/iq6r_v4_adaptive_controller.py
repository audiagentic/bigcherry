"""Fail-closed adaptive IQ6R controller.

Requires a clean handoff from the original micro controller; retains the
micro exclusion lock for its entire lifetime. Each conversion has its own
exclusive GPU lease and a verified sealed RAD2 receipt.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import secrets
import subprocess
import sys
from pathlib import Path

from iq6r_v4_controller import WORK, stage_environment

sys.path.insert(0, str(WORK / "pydeps"))
from iq6r_v4_adaptive import LOCK, MICRO_LOCK, STATE, next_job, source_recipes
from iq6r_v4_staged_convert import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-timeout", type=int, default=10800)
    args = parser.parse_args()
    if not 300 <= args.stage_timeout <= 10800:
        parser.error("stage timeout outside 300..10800")
    with LOCK.open("a+") as lock, MICRO_LOCK.open("a+") as micro_lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(micro_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("micro controller still active; no handoff") from exc
        token = secrets.token_hex(32)
        state = {
            "status": "starting",
            "pid": os.getpid(),
            "controller_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "production_promotion": False,
            "quality_passed": False,
            "tp3_passed": False,
        }
        sources = source_recipes()
        # Preflight the checkpoint chain before writing the controller status.
        next_job(sources)
        write_json(STATE, state)
        try:
            while True:
                job = next_job(sources)
                state.update({
                    "next_stage": job["stage"],
                    "gptq_before": job["gptq_before"],
                    "gptq_after": job["gptq_after"],
                    "batch": job["batch"],
                })
                if job["done"]:
                    state["status"] = "fully_sealed_unqualified"
                    write_json(STATE, state)
                    return 0
                state["status"] = "queued_for_exclusive_lease"
                write_json(STATE, state)
                cmd = [
                    "agent-lease", "acquire", "gpu", "--wait",
                    "--est", "2h", "--ttl", "4h",
                    "--purpose", f"IQ6R adaptive checkpoint {job['stage']:04d}",
                    "--", sys.executable, "-u",
                    str(Path(__file__).with_name("iq6r_v4_adaptive.py")),
                    "--managed", "--stage-timeout", str(args.stage_timeout),
                ]
                env = stage_environment()
                env["BC_IQ6R_ADAPTIVE_TOKEN"] = token
                rc = subprocess.run(cmd, env=env, check=False).returncode
                if rc:
                    raise RuntimeError(f"adaptive stage {job['stage']} failed ({rc}); no retry")
                after = next_job(sources)
                if after["gptq_before"] != job["gptq_after"]:
                    raise RuntimeError(f"adaptive stage {job['stage']} did not seal expected receipt")
                state["status"] = "checkpoint_verified"
                state["last_sealed_stage"] = job["stage"]
                write_json(STATE, state)
        except BaseException as exc:
            state["status"] = "blocked_no_retry"
            state["error"] = str(exc)
            write_json(STATE, state)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
