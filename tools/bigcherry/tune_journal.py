"""Current-schema append-only tuning journal and deterministic compactor (HI48)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1
MODES = {"buffered", "batch", "durable_each"}


class JournalError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def checksum(event_without_checksum: dict[str, Any]) -> str:
    return hashlib.blake2b(
        canonical(event_without_checksum), digest_size=16,
        person=b"bc-journal-v1",
    ).hexdigest()


def _checked_event(event: dict[str, Any]) -> dict[str, Any]:
    result = dict(event)
    result["checksum"] = checksum(result)
    return result


def atomic_write(path: Path, data: bytes) -> None:
    """Same-directory temp file, flush, fsync, atomic rename. Never truncates
    the destination before the replacement is fully durable."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class JournalWriter:
    """Append-only writer. One self-contained JSON event per line, each
    independently checksummed. `durable_each` fsyncs after every event;
    `batch(N)` fsyncs every N events; `buffered` leaves flushing to the OS.
    """

    def __init__(
            self, path: Path, experiment_id: str, source_revision: str,
            manifest_hash: str, hardware_digest: str, *,
            durability_mode: Literal["buffered", "batch", "durable_each"] = "durable_each",
            batch_size: int = 1, storage_kind: str = "local",
    ) -> None:
        if durability_mode not in MODES:
            raise JournalError(f"unknown durability_mode {durability_mode!r}")
        self.path = path
        self.experiment_id = experiment_id
        self.durability_mode = durability_mode
        self.batch_size = batch_size
        self.storage_kind = storage_kind
        self._sequence = 0
        self._unflushed = 0
        self._closed = False
        # Exclusive-create: refuses to append to or truncate a journal left
        # by a prior run. A killed run's journal is evidence, not scratch.
        self._handle = open(path, "xb")

        self.append("start", {
            "batch_size": self.batch_size,
            "durability_mode": self.durability_mode,
            "hardware_digest": hardware_digest,
            "manifest_hash": manifest_hash,
            "source_revision": source_revision,
            "storage_kind": self.storage_kind,
            "durability_scope": (
                "process" if self.durability_mode == "buffered" else
                "os_cache" if self.storage_kind == "network" else
                "storage_claimed"
            ),
        })

    def append(self, kind: str, payload: Any) -> int:
        if self._closed:
            raise JournalError("journal is closed")
        self._sequence += 1
        event = _checked_event({
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "sequence": self._sequence,
            "kind": kind,
            "payload": payload,
        })
        self._handle.write(canonical(event) + b"\n")
        self._unflushed += 1
        if (self.durability_mode == "durable_each" or
                self.durability_mode == "batch" and self._unflushed >= self.batch_size):
            self.acknowledge()
        return self._sequence

    def result(self, record: dict[str, Any]) -> int:
        if record.get("kind") != "result" or not record.get("dispatch"):
            raise JournalError("journal result must be a current measurements result")
        return self.append("result", record)

    def acknowledge(self) -> int:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._unflushed = 0
        return self._sequence

    def complete(self, summary: dict[str, Any] | None = None) -> None:
        self.append("complete", summary or {})
        self.acknowledge()
        self._handle.close()
        self._closed = True

    def interrupt(self, reason: str) -> None:
        self.append("interrupted", {"reason": reason})
        self.acknowledge()
        self._handle.close()
        self._closed = True


def read_current(path: Path, *, recover_tail: bool = True) -> dict[str, Any]:
    """Reads and validates every complete, checksummed event. On a torn or
    corrupt final line (only, and only if `recover_tail`), stops there and
    reports the prefix as recovered rather than raising -- a killed process
    leaves exactly this shape, and it is the expected, handled case, not an
    error.
    """
    data = path.read_bytes()
    # keepends=True: each line carries its own terminator (or none, for a
    # torn final write), so completeness is checkable per-line directly
    # rather than inferred from position in a pre-split array.
    lines = data.splitlines(keepends=True)

    events: list[dict[str, Any]] = []
    corrupt_tail = False
    corruption: str | None = None
    expected_sequence = 1
    for index, raw_line in enumerate(lines):
        complete_line = raw_line.endswith((b"\n", b"\r"))
        raw = raw_line.rstrip(b"\r\n")
        if not raw:
            continue
        if not complete_line and index == len(lines) - 1 and recover_tail:
            corrupt_tail = True
            corruption = f"truncated record at sequence {expected_sequence}"
            break
        try:
            event = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if recover_tail and index == len(lines) - 1:
                corrupt_tail = True
                corruption = f"malformed tail at sequence {expected_sequence}: {exc}"
                break
            raise JournalError(f"journal line {index + 1} is malformed") from exc
        if event.get("schema_version") != SCHEMA_VERSION:
            raise JournalError("rerun_or_external_conversion_required: journal schema")
        stored = event.pop("checksum", None)
        if stored != checksum(event):
            if recover_tail and index == len(lines) - 1:
                corrupt_tail = True
                corruption = f"checksum mismatch at sequence {event.get('sequence')}"
                break
            raise JournalError(f"journal checksum mismatch at sequence {event.get('sequence')}")
        event["checksum"] = stored
        if event.get("sequence") != expected_sequence:
            raise JournalError(
                f"journal sequence gap/duplicate: expected {expected_sequence}, got {event.get('sequence')}")
        expected_sequence += 1
        events.append(event)
    if not events or events[0].get("kind") != "start":
        raise JournalError("current journal start event missing")
    experiment_ids = {event.get("experiment_id") for event in events}
    if len(experiment_ids) != 1:
        raise JournalError("journal mixes experiment identities")
    terminal = events[-1].get("kind") if events else None
    # A trailing "attempt" (HI48 diagnostic extension: hip-autotune-tuner.cu's
    # trace_launch_attempt, opt-in via GGML_HIP_TUNE_TRACE_ATTEMPTS) means the
    # process stopped executing C++ while a specific candidate's GPU work was
    # in flight -- every "result" before it is still exactly as recoverable
    # as an ordinary interrupted run (compact() already skips non-"result"
    # events), so it belongs in can_compact alongside them. It is surfaced
    # separately as last_attempt rather than folded into `interrupted`,
    # which HI48 already defines narrowly (explicit interrupt or corrupt
    # tail) -- an attempt-terminated run is a stronger, different signal
    # (specifically: this candidate's kernel is the leading suspect for
    # whatever stopped the process) and deserves its own field rather than
    # overloading an established one.
    last_attempt = None
    if terminal == "attempt":
        payload = events[-1].get("payload")
        if isinstance(payload, str):
            try:
                last_attempt = json.loads(payload)
            except json.JSONDecodeError:
                last_attempt = {"raw": payload}
        else:
            last_attempt = payload
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": events[0]["experiment_id"],
        "events": events,
        "last_durable_sequence": len(events),
        "corrupt_tail": corrupt_tail,
        "corruption": corruption,
        "complete": terminal == "complete" and not corrupt_tail,
        "interrupted": terminal == "interrupted" or corrupt_tail,
        "last_attempt": last_attempt,
        "can_compact": bool(events) and terminal in {"complete", "interrupted", "result", "start", "attempt"},
    }


def compact(path: Path, output: Path, header: dict[str, Any]) -> dict[str, Any]:
    journal = read_current(path)
    if header.get("kind") != "header":
        raise JournalError("compaction requires a current measurements header")
    # The durable start event is the authoritative provenance for C++-originated
    # journals.  Preserve it when callers use the minimal CLI header, otherwise
    # a compacted run cannot pass replay-cache provenance validation.
    compacted_header = dict(header)
    start_payload = journal["events"][0].get("payload")
    if isinstance(start_payload, dict):
        for field in ("source_revision", "manifest_hash", "hardware_digest"):
            if field not in compacted_header and field in start_payload:
                compacted_header[field] = start_payload[field]
    selected: dict[str, tuple[int, dict[str, Any]]] = {}
    for event in journal["events"]:
        if event["kind"] != "result":
            continue
        record = event["payload"]
        if isinstance(record, str):
            # A C++-originated journal (hip-autotune-journal.cpp) stores a
            # result's payload as an opaque JSON string rather than a nested
            # object -- Python's canonical() would otherwise have to
            # recursively sort every key of the tuner's ~80-field result
            # JSON for the envelope checksum to verify, which the C++ side
            # does not attempt. One extra parse here recovers the same
            # dict a Python-authored journal already provides directly.
            try:
                record = json.loads(record)
            except json.JSONDecodeError as exc:
                raise JournalError(
                    f"result payload at sequence {event['sequence']} is not valid JSON") from exc
        # Early C++ journal summaries omitted the fields that identify an
        # unchanged native outcome.  The stable native spelling is
        # unambiguous; recover only that case.  Every other missing native or
        # promotion state remains fail-closed in replay_cache.py.
        winner = record.get("winner")
        if (isinstance(winner, str) and winner.endswith(":native:v1") and
                "native" not in record and "promotion_status" not in record):
            record = dict(record)
            record["native"] = winner
            record["promotion_status"] = "native"
        dispatch = record.get("dispatch")
        if not isinstance(dispatch, str):
            raise JournalError("result event lacks dispatch identity")
        generation = int(record.get("generation", 1))
        current = selected.get(dispatch)
        rank = (generation, event["sequence"])
        if current is None or rank > (int(current[1].get("generation", 1)), current[0]):
            selected[dispatch] = (event["sequence"], record)
    lines = [canonical(compacted_header)]
    lines.extend(canonical(selected[key][1]) for key in sorted(selected))
    data = b"\n".join(lines) + b"\n"
    atomic_write(output, data)
    return {
        "content_hash": hashlib.blake2b(data, digest_size=16).hexdigest(),
        "output": str(output),
        "hypotheses": len(selected),
        "promoted": len(selected),
        "schema_version": SCHEMA_VERSION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry tune-journal")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    status = sub.add_parser("status")
    status.add_argument("journal", type=Path)

    compact_cmd = sub.add_parser("compact")
    compact_cmd.add_argument("journal", type=Path)
    compact_cmd.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.subcommand == "status":
        try:
            result = read_current(args.journal)
        except JournalError as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.subcommand == "compact":
        try:
            result = compact(args.journal, args.output, {"kind": "header", "schema_version": SCHEMA_VERSION})
        except JournalError as exc:
            print(f"invalid: {exc}")
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
