"""Filesystem-backed durable authority for BigCherry jobs."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..core.host_lock import HostFileLock
from ..tuning.journal import atomic_write
from .model import canonical_bytes, digest


class StoreError(RuntimeError):
    pass


class IdempotencyConflict(StoreError):
    pass


def _write_json(path: Path, value: dict[str, Any], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    if path.exists() and not replace:
        existing = json.loads(path.read_text(encoding="ascii"))
        if existing != value:
            raise StoreError(f"immutable record already exists with different content: {path}")
        return
    atomic_write(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(f"cannot read durable record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StoreError(f"durable record is not an object: {path}")
    return value


def _checksum(value: dict[str, Any]) -> str:
    return hashlib.blake2b(canonical_bytes(value), digest_size=20, person=b"bc-job-event-v1").hexdigest()


@dataclass(frozen=True)
class EventRecord:
    seq: int
    event_id: str
    ts_ns: int
    kind: str
    severity: str
    run_id: str | None
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "ts_ns": self.ts_ns,
            "kind": self.kind,
            "severity": self.severity,
            "run_id": self.run_id,
            "data": self.data,
        }


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("requests", "batches", "runs", "inbox/pending", "inbox/processing", "inbox/accepted", "inbox/rejected"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @property
    def event_path(self) -> Path:
        return self.root / "events.jsonl"

    def accept_idempotency(self, key: str, request_digest: str, batch_id: str) -> dict[str, Any]:
        if not key.strip():
            raise ValueError("idempotency key is required")
        key_hash = digest({"key": key}, person=b"bc-idempotency")
        path = self.root / "requests" / f"{key_hash}.json"
        record = {"key": key, "request_digest": request_digest, "batch_id": batch_id}
        with HostFileLock(self.root / "requests.lock"):
            if path.exists():
                existing = _read_json(path)
                if existing.get("request_digest") != request_digest:
                    raise IdempotencyConflict("idempotency key was already used for a different request")
                return existing
            _write_json(path, record)
        return record

    def create_batch(self, batch_id: str, record: dict[str, Any]) -> None:
        _write_json(self.root / "batches" / batch_id / "batch.json", record)

    def batch(self, batch_id: str) -> dict[str, Any]:
        return _read_json(self.root / "batches" / batch_id / "batch.json")

    def create_run(self, run_id: str, record: dict[str, Any]) -> None:
        _write_json(self.root / "runs" / run_id / "intent.json", record)

    def run(self, run_id: str) -> dict[str, Any]:
        return _read_json(self.root / "runs" / run_id / "intent.json")

    def list_runs(self) -> tuple[dict[str, Any], ...]:
        values: list[dict[str, Any]] = []
        for path in sorted((self.root / "runs").glob("*/intent.json")):
            values.append(_read_json(path))
        return tuple(values)

    def next_attempt(self, run_id: str) -> int:
        attempts = self.root / "runs" / run_id / "attempts"
        if not attempts.exists():
            return 1
        numbers = [int(path.name) for path in attempts.iterdir() if path.is_dir() and path.name.isdigit()]
        return max(numbers, default=0) + 1

    def attempt_root(self, run_id: str, attempt: int) -> Path:
        return self.root / "runs" / run_id / "attempts" / f"{attempt:03d}"

    def create_attempt(self, run_id: str, attempt: int, record: dict[str, Any]) -> Path:
        root = self.attempt_root(run_id, attempt)
        _write_json(root / "attempt.json", record)
        return root

    def attempt(self, run_id: str, attempt: int) -> dict[str, Any]:
        return _read_json(self.attempt_root(run_id, attempt) / "attempt.json")

    def write_submission_intent(self, run_id: str, attempt: int, record: dict[str, Any]) -> None:
        _write_json(self.attempt_root(run_id, attempt) / "submission-intent.json", record)

    def write_submission(self, run_id: str, attempt: int, record: dict[str, Any]) -> None:
        _write_json(self.attempt_root(run_id, attempt) / "submission.json", record)

    def write_result(self, run_id: str, attempt: int, record: dict[str, Any]) -> None:
        _write_json(self.attempt_root(run_id, attempt) / "result.json", record)

    def read_optional(self, path: Path) -> dict[str, Any] | None:
        return _read_json(path) if path.is_file() else None

    def enqueue(self, batch_id: str, run_id: str) -> Path:
        receipt = {"batch_id": batch_id, "run_id": run_id}
        path = self.root / "inbox" / "pending" / f"{run_id}.json"
        _write_json(path, receipt)
        return path

    def pending(self) -> tuple[Path, ...]:
        return tuple(sorted((self.root / "inbox" / "pending").glob("*.json")))

    def claim_pending(self, path: Path) -> Path:
        target = self.root / "inbox" / "processing" / path.name
        try:
            os.replace(path, target)
        except FileNotFoundError as exc:
            raise StoreError(f"pending receipt was already claimed: {path}") from exc
        return target

    def finish_receipt(self, processing: Path, *, accepted: bool) -> None:
        dest = self.root / "inbox" / ("accepted" if accepted else "rejected") / processing.name
        os.replace(processing, dest)

    def append_event(
        self,
        *,
        kind: str,
        data: dict[str, Any] | None = None,
        run_id: str | None = None,
        severity: str = "record",
    ) -> EventRecord:
        import uuid

        if severity not in {"record", "notify", "wake"}:
            raise ValueError("invalid event severity")
        lock = HostFileLock(self.root / "events.lock")
        with lock:
            events, valid_bytes = self._read_events_with_offset()
            if self.event_path.exists() and self.event_path.stat().st_size != valid_bytes:
                with self.event_path.open("r+b") as handle:
                    handle.truncate(valid_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            seq = events[-1].seq + 1 if events else 1
            record = EventRecord(seq, str(uuid.uuid4()), time.time_ns(), kind, severity, run_id, data or {})
            body = record.to_dict()
            envelope = dict(body, checksum=_checksum(body))
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_path.open("ab") as handle:
                handle.write(canonical_bytes(envelope) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            return record

    def read_events(self, *, after: int = 0) -> tuple[EventRecord, ...]:
        events, _ = self._read_events_with_offset()
        return tuple(event for event in events if event.seq > after)

    def _read_events_with_offset(self) -> tuple[list[EventRecord], int]:
        if not self.event_path.exists():
            return [], 0
        events: list[EventRecord] = []
        valid_bytes = 0
        raw = self.event_path.read_bytes()
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.endswith(b"\n"):
                break
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                raise StoreError("corrupt non-tail event record")
            if not isinstance(envelope, dict):
                raise StoreError("event record must be an object")
            checksum = envelope.pop("checksum", None)
            if checksum != _checksum(envelope):
                if index == len(lines) - 1:
                    break
                raise StoreError("event checksum mismatch before tail")
            expected = len(events) + 1
            if envelope.get("seq") != expected:
                raise StoreError(f"event sequence mismatch: expected {expected}, got {envelope.get('seq')}")
            events.append(EventRecord(
                seq=int(envelope["seq"]),
                event_id=str(envelope["event_id"]),
                ts_ns=int(envelope["ts_ns"]),
                kind=str(envelope["kind"]),
                severity=str(envelope["severity"]),
                run_id=None if envelope.get("run_id") is None else str(envelope["run_id"]),
                data=dict(envelope.get("data") or {}),
            ))
            valid_bytes += len(line)
        return events, valid_bytes
