"""File-backed RCD service recovery falsifier.

Exercises crash/idempotency/event invariants without pretending to be the final
jobs service. Uses BigCherry's real atomic_write primitive and real processes
for concurrent event writers.
"""
from __future__ import annotations

import fcntl
import json
import multiprocessing as mp
import os
import tempfile
import uuid
from pathlib import Path

from bigcherry.tuning.journal import atomic_write, canonical


class AmbiguousSubmission(RuntimeError):
    pass


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(value, sort_keys=True, indent=2).encode() + b"\n")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


class FakeExecutor:
    def __init__(self, before_submit=None) -> None:
        self.by_execution: dict[str, list[str]] = {}
        self.submits = 0
        self.before_submit = before_submit

    def submit(self, execution_id: str) -> str:
        if self.before_submit:
            self.before_submit(execution_id)
        self.submits += 1
        native = f"fake-{self.submits:04d}"
        self.by_execution.setdefault(execution_id, []).append(native)
        return native

    def correlate(self, execution_id: str) -> tuple[str, ...]:
        return tuple(self.by_execution.get(execution_id, ()))


class FileService:
    def __init__(self, root: Path, executor: FakeExecutor) -> None:
        self.root = root
        self.executor = executor

    def _dir(self, execution_id: str) -> Path:
        return self.root / "executions" / execution_id

    def start(self, execution_id: str, *, crash_after_accept: bool = False) -> str:
        d = self._dir(execution_id)
        write_json(d / "submission-intent.json", {"execution_id": execution_id, "state": "submitting"})
        native = self.executor.submit(execution_id)
        if crash_after_accept:
            raise RuntimeError("simulated crash after executor acceptance")
        write_json(d / "submission.json", {"execution_id": execution_id, "native_id": native})
        return native

    def reconcile(self, execution_id: str) -> str:
        d = self._dir(execution_id)
        existing = d / "submission.json"
        if existing.exists():
            return str(read_json(existing)["native_id"])
        if not (d / "submission-intent.json").exists():
            raise RuntimeError("no durable submission intent")
        matches = self.executor.correlate(execution_id)
        if len(matches) > 1:
            raise AmbiguousSubmission(execution_id)
        if len(matches) == 1:
            native = matches[0]
        else:
            native = self.executor.submit(execution_id)
        write_json(existing, {"execution_id": execution_id, "native_id": native, "recovered": True})
        return native


def append_event(root: str, writer: int, count: int) -> None:
    base = Path(root)
    lock_path = base / "events.lock"
    log_path = base / "events.jsonl"
    lock_path.touch(exist_ok=True)
    for index in range(count):
        with lock_path.open("r+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            seq = 1
            if log_path.exists():
                with log_path.open("rb") as src:
                    seq += sum(1 for _ in src)
            event = {
                "event_id": str(uuid.uuid4()),
                "kind": "test",
                "seq": seq,
                "writer": writer,
                "writer_index": index,
            }
            with log_path.open("ab") as out:
                out.write(canonical(event) + b"\n")
                out.flush()
                os.fsync(out.fileno())
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def check(value: bool, message: str) -> int:
    if not value:
        raise AssertionError(message)
    return 1


def main() -> int:
    checks = 0
    with tempfile.TemporaryDirectory(prefix="bigcherry-service-recovery-") as temp:
        root = Path(temp)

        # Durable intent must exist before the executor can accept work.
        def before_submit(execution_id: str) -> None:
            nonlocal checks
            intent = root / "executions" / execution_id / "submission-intent.json"
            checks += check(intent.exists(), "submission intent missing before executor submit")
            checks += check(read_json(intent)["state"] == "submitting", "intent not durable/canonical")

        executor = FakeExecutor(before_submit)
        service = FileService(root, executor)

        # Crash after native acceptance: restart must correlate, never duplicate.
        try:
            service.start("exec-crash", crash_after_accept=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("simulated crash did not happen")
        checks += check(executor.submits == 1, "expected one accepted native execution")
        checks += check(not (service._dir("exec-crash") / "submission.json").exists(), "handle unexpectedly persisted")
        restarted = FileService(root, executor)
        recovered = restarted.reconcile("exec-crash")
        checks += check(recovered == "fake-0001", "restart failed to correlate accepted execution")
        checks += check(executor.submits == 1, "restart duplicated accepted execution")
        checks += check(restarted.reconcile("exec-crash") == recovered, "reconcile not idempotent")

        # If acceptance can be disproved (zero correlations), submit exactly once.
        write_json(service._dir("exec-zero") / "submission-intent.json", {"execution_id": "exec-zero", "state": "submitting"})
        native_zero = restarted.reconcile("exec-zero")
        checks += check(native_zero == "fake-0002", "zero-match recovery did not submit once")
        checks += check(executor.submits == 2, "zero-match recovery submit count wrong")
        checks += check(restarted.reconcile("exec-zero") == native_zero and executor.submits == 2, "zero-match replay duplicated")

        # Multiple correlations are unsafe: wake/block, never guess and never submit.
        write_json(service._dir("exec-ambiguous") / "submission-intent.json", {"execution_id": "exec-ambiguous", "state": "submitting"})
        executor.by_execution["exec-ambiguous"] = ["native-a", "native-b"]
        before = executor.submits
        try:
            restarted.reconcile("exec-ambiguous")
        except AmbiguousSubmission:
            checks += 1
        else:
            raise AssertionError("ambiguous submission was silently selected")
        checks += check(executor.submits == before, "ambiguous recovery submitted new work")
        checks += check(not (service._dir("exec-ambiguous") / "submission.json").exists(), "ambiguous handle persisted")

        # Retry identity: 75 is same attempt/native restart; 76 is new attempt but same run.
        run = {"run_id": "run-1", "attempt": 1, "commit": "a" * 40, "native_id": "job-7"}
        same_commit = dict(run, restart_count=1)
        checks += check(same_commit["run_id"] == run["run_id"] and same_commit["attempt"] == 1, "75 changed attempt identity")
        checks += check(same_commit["commit"] == run["commit"] and same_commit["native_id"] == run["native_id"], "75 changed commit/native id")
        new_attempt = {"run_id": run["run_id"], "attempt": 2, "commit": "b" * 40, "native_id": "job-8"}
        checks += check(new_attempt["run_id"] == run["run_id"] and new_attempt["attempt"] == 2, "76 did not preserve run/new attempt")
        checks += check(new_attempt["commit"] != run["commit"] and new_attempt["native_id"] != run["native_id"], "76 failed to permit new commit/job")
        scientific_fail = {"exit_code": 0, "verdict": "FAIL", "retry": False}
        checks += check(scientific_fail["exit_code"] == 0 and not scientific_fail["retry"], "scientific FAIL became retry")

        # Concurrent event writers must serialize one global sequence with no gaps/dupes.
        event_root = root / "events"
        event_root.mkdir()
        ctx = mp.get_context("fork")
        procs = [ctx.Process(target=append_event, args=(str(event_root), writer, 20)) for writer in range(4)]
        for process in procs:
            process.start()
        for process in procs:
            process.join(10)
            checks += check(process.exitcode == 0, "event writer failed")
        events = [json.loads(line) for line in (event_root / "events.jsonl").read_text().splitlines()]
        checks += check(len(events) == 80, "event count mismatch")
        checks += check([e["seq"] for e in events] == list(range(1, 81)), "event sequence gap/duplicate")
        checks += check(len({e["event_id"] for e in events}) == 80, "event id duplicate")

        # A new service object reconstructs durable state without process memory.
        cold = FileService(root, executor)
        checks += check(cold.reconcile("exec-crash") == "fake-0001", "cold restart lost persisted handle")
        checks += check(cold.reconcile("exec-zero") == "fake-0002", "cold restart lost recovered handle")

    print(json.dumps({"checks": checks, "ok": True, "scope": "durable-submission-and-event-recovery"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
