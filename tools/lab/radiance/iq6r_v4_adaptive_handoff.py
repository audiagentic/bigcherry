"""Controlled handoff from a live micro stage to adaptive checkpoints.

Stop ONLY the idle controller parent while its already-launched child finishes.
Never kill an in-progress GPU job. Validate the sealed output and await the
child's exit before retiring the stopped parent and exec'ing the new controller.
The parent resumes on failures or an interrupted handoff.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

from iq6r_v4_controller import WORK
from iq6r_v4_micro_convert import MICRO_ROOT, sha

OLD = WORK / "conversion-controller-micro.json"
STATUS = WORK / "conversion-adaptive-handoff.json"
STAGE = 3
MAX_WAIT = 10800


def worker_alive(stage: int) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if not any(arg.endswith(b"/iq6r_v4_micro_convert.py") for arg in args):
            continue
        if b"--stage" in args and args[args.index(b"--stage") + 1] == str(stage).encode():
            return True
    return False


def write_status(status: str, **kwargs: object) -> None:
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps({"status": status, **kwargs}, indent=2) + "\n")
    temp.replace(STATUS)


def stage_sealed(stage: int) -> bool:
    stem = MICRO_ROOT / f"micro-{stage:04d}"
    receipt = stem.with_suffix(".json")
    file = stem.with_suffix(".rad")
    if not receipt.is_file() or not file.is_file():
        return False
    record = json.loads(receipt.read_text())
    if record.get("stage") != stage or record.get("gptq_cumulative") != stage + 3:
        raise RuntimeError("sealed stage receipt identity differs")
    if file.stat().st_size != record["bytes"] or sha(file) != record["sha256"]:
        raise RuntimeError("sealed stage content checksum mismatch")
    return True


def parent_alive(pid: int) -> bool:
    try:
        cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return False
    return b"iq6r_v4_micro_controller.py" in cmd


def main() -> int:
    state = json.loads(OLD.read_text())
    pid = int(state["pid"])
    if (not parent_alive(pid) or state.get("next_stage") != STAGE
            or state.get("status") != "queued_for_exclusive_lease"
            or not worker_alive(STAGE)):
        raise RuntimeError("stage-3 micro conversion not actively owned; refusing handoff")
    if (MICRO_ROOT / "micro-0004.recipe").exists():
        raise RuntimeError("next micro stage already started")
    # Exact PID only. The separate GPU worker is not stopped.
    os.kill(pid, signal.SIGSTOP)
    write_status("parent_paused_waiting_for_stage3", parent_pid=pid)
    parked = True
    try:
        deadline = time.monotonic() + MAX_WAIT
        while time.monotonic() < deadline:
            if (MICRO_ROOT / "micro-0004.recipe").exists():
                raise RuntimeError("race: micro stage 4 started; cannot hand off")
            if not parent_alive(pid):
                raise RuntimeError("micro controller disappeared before handoff")
            if not worker_alive(STAGE):
                if not stage_sealed(STAGE):
                    raise RuntimeError("micro stage 3 exited without a sealed receipt")
                break
            time.sleep(4)
        else:
            raise RuntimeError("micro stage 3 did not finish before handoff timeout")
        # The worker has exited and its sealed receipt is SHA-verified.
        # Only now retire the paused controller before it can launch stage 4.
        write_status("stage3_verified_retiring_old_parent", parent_pid=pid)
        os.kill(pid, signal.SIGKILL)
        parked = False
        deadline = time.monotonic() + 30
        while parent_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.25)
        if parent_alive(pid):
            raise RuntimeError("old controller still live; refusing adaptive start")
        write_status("old_parent_retired_starting_adaptive")
        controller = Path(__file__).with_name("iq6r_v4_adaptive_controller.py")
        os.execv(sys.executable, [sys.executable, "-u", str(controller)])
    finally:
        if parked and parent_alive(pid):
            os.kill(pid, signal.SIGCONT)
            write_status("aborted_old_parent_resumed", parent_pid=pid)


if __name__ == "__main__":
    raise SystemExit(main())
