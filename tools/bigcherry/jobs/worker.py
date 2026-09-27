"""Restricted JSON worker endpoint used by RemoteExecutor over SSH."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from .executor import ExecutionHandle, ExecutorControl
from .local import LocalExecutor
from .remote import request_from_dict


def _payload() -> dict[str, object]:
    raw = sys.stdin.read()
    value = json.loads(raw or "{}")
    if not isinstance(value, dict):
        raise ValueError("worker payload must be an object")
    return value


def _handle(executor: LocalExecutor, value: dict[str, object]) -> ExecutionHandle:
    return ExecutionHandle(executor.name, str(value["native_id"]), str(value["execution_id"]))


def run(action: str, payload: dict[str, object], executor: LocalExecutor) -> dict[str, object]:
    if action == "submit":
        request = request_from_dict(dict(payload["request"]))
        handle = executor.submit(request)
        return {"native_id": handle.native_id}
    if action == "correlate":
        return {"native_ids": [handle.native_id for handle in executor.correlate(str(payload["execution_id"]))]}
    if action == "capabilities":
        return {"executor": executor.name, "platform": os.name, "protocol": "bigcherry.remote-worker.v1"}
    handle = _handle(executor, payload)
    if action == "status":
        status = executor.status(handle)
        return {"state": status.state.value, "reason": status.reason, "native_state": status.native_state}
    if action == "cancel":
        executor.cancel(handle); return {"ok": True}
    if action == "control":
        executor.control(handle, ExecutorControl(str(payload["action"]))); return {"ok": True}
    if action == "allocation":
        allocation = executor.allocation(handle)
        return {"allocation": None if allocation is None else dataclasses.asdict(allocation)}
    if action == "events":
        after = payload.get("after")
        events = executor.events(handle, after=None if after is None else int(after))
        return {"events": [dataclasses.asdict(event) for event in events]}
    raise ValueError(f"unsupported worker action: {action}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry jobs-worker")
    parser.add_argument("action", choices=("submit", "correlate", "status", "cancel", "control", "allocation", "events", "capabilities"))
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root or Path(os.environ.get("BIGCHERRY_JOBS_WORK_ROOT", Path.home() / ".bigcherry-jobs-worker"))
    executor = LocalExecutor(root / "local-executor")
    try:
        result = run(args.action, _payload(), executor)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
