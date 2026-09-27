"""Human/agent CLI over the scheduler-neutral JobService API."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import tomllib
from pathlib import Path

from ..core.context import ProjectContext
from ..hardware.inventory import InventoryCatalog
from ..jobs.model import batch_from_mapping
from ..jobs.registry import load_executor_registry
from ..jobs.service import JobService
from ..jobs.store import RunStore
from ..jobs.workspace import GitWorkspaceManager


def _json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _load_document(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    if path.suffix.lower() == ".toml":
        value = tomllib.loads(raw.decode("utf-8"))
    else:
        value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("job specification must be an object/table")
    # TOML examples may wrap the request in [batch].
    if isinstance(value.get("batch"), dict):
        value = dict(value["batch"])
    return value


def build_service() -> tuple[JobService, object]:
    context = ProjectContext.resolve()
    jobs_root = Path(os.environ.get("BIGCHERRY_JOBS_ROOT", context.work_root / "jobs")).resolve()
    registry_path = Path(os.environ.get("BIGCHERRY_EXECUTORS_CONFIG", context.project_root / "config" / "jobs" / "executors.toml")).resolve()
    hardware_root = Path(os.environ.get("BIGCHERRY_HARDWARE_ROOT", context.work_root / "hardware")).resolve()
    registry = load_executor_registry(registry_path, state_root=jobs_root)
    catalog = InventoryCatalog(hardware_root)
    service = JobService(
        store=RunStore(jobs_root),
        executors=registry.executors,
        workspace_manager=GitWorkspaceManager(context.project_root, jobs_root / "worktrees"),
        inventory_loader=catalog.load,
        shared_root=context.work_root,
    )
    return service, registry


def _event_dict(event) -> dict[str, object]:
    return event.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry jobs")
    sub = parser.add_subparsers(dest="action", required=True)

    for name in ("plan", "validate", "submit"):
        cmd = sub.add_parser(name)
        cmd.add_argument("spec", type=Path)
        if name == "submit":
            cmd.add_argument("--idempotency-key", required=True)
            cmd.add_argument("--actor", default=os.environ.get("USER", os.environ.get("USERNAME", "unknown")))

    sub.add_parser("ingest").add_argument("--once", action="store_true", default=True)
    sub.add_parser("list")
    show = sub.add_parser("show"); show.add_argument("run_id")
    status = sub.add_parser("status"); status.add_argument("run_id", nargs="?")

    events = sub.add_parser("events")
    events.add_argument("--after", type=int, default=0)
    events.add_argument("--run-id", default=None)
    events.add_argument("--wake-only", action="store_true")
    events.add_argument("--follow", action="store_true")

    logs = sub.add_parser("logs")
    logs.add_argument("run_id")
    logs.add_argument("--stream", choices=("stdout", "stderr"), default="stdout")
    logs.add_argument("--offset", type=int, default=0)
    logs.add_argument("--limit", type=int, default=1024 * 1024)

    artifacts = sub.add_parser("artifacts"); artifacts.add_argument("run_id")
    review = sub.add_parser("review"); review.add_argument("series_id")

    for name in ("disable", "enable", "cancel", "hold", "release"):
        cmd = sub.add_parser(name); cmd.add_argument("run_id")

    retry = sub.add_parser("retry")
    retry.add_argument("run_id")
    mode = retry.add_mutually_exclusive_group(required=True)
    mode.add_argument("--latest", action="store_true")
    mode.add_argument("--same-commit", action="store_true")

    sub.add_parser("pause")
    sub.add_parser("resume")
    executors = sub.add_parser("executors")
    executors.add_argument("executor_action", choices=("list",))

    args = parser.parse_args(argv)
    try:
        service, registry = build_service()
        if args.action in {"plan", "validate", "submit"}:
            batch = batch_from_mapping(_load_document(args.spec))
            plan = service.plan_batch(batch)
            if args.action in {"plan", "validate"}:
                _json({"ok": True, "plan": plan})
                return 0
            _json(service.submit_batch(batch, idempotency_key=args.idempotency_key, actor=args.actor))
            return 0
        if args.action == "ingest":
            _json(service.ingest_once()); return 0
        if args.action == "list":
            _json({"jobs": list(service.list_status())}); return 0
        if args.action == "show":
            _json({"intent": service.store.run(args.run_id), "status": service.status(args.run_id), "artifacts": list(service.artifacts(args.run_id))}); return 0
        if args.action == "status":
            if args.run_id:
                _json(service.status(args.run_id))
            else:
                events_now = service.store.read_events()
                _json({"paused": bool(service.store.service_control().get("paused")), "last_event_seq": events_now[-1].seq if events_now else 0, "jobs": list(service.list_status())})
            return 0
        if args.action == "events":
            after = args.after
            while True:
                emitted = False
                for event in service.store.read_events(after=after):
                    after = event.seq
                    if args.run_id and event.run_id != args.run_id:
                        continue
                    if args.wake_only and event.severity != "wake":
                        continue
                    _json(_event_dict(event)); emitted = True
                if not args.follow:
                    return 0
                if emitted:
                    sys.stdout.flush()
                time.sleep(0.5)
        if args.action == "logs":
            _json(service.logs(args.run_id, stream=args.stream, offset=args.offset, limit=args.limit)); return 0
        if args.action == "artifacts":
            _json({"run_id": args.run_id, "artifacts": list(service.artifacts(args.run_id))}); return 0
        if args.action == "review":
            _json(service.review(args.series_id)); return 0
        if args.action in {"disable", "enable", "cancel", "hold", "release"}:
            _json(service.control_run(args.run_id, args.action)); return 0
        if args.action == "retry":
            service.retry(args.run_id, latest=args.latest)
            _json({"run_id": args.run_id, "retry_requested": True, "latest": args.latest}); return 0
        if args.action in {"pause", "resume"}:
            _json(service.pause(args.action == "pause")); return 0
        if args.action == "executors":
            _json({"executors": list(registry.describe())}); return 0
        raise AssertionError(args.action)
    except Exception as exc:
        _json({"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
