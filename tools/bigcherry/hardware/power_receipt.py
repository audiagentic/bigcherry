"""PWR07/PWR09: durable JSON persistence for power receipts and transaction journals.

``write_json_atomic`` is the one place that writes either: temporary file in the target directory, file fsync, atomic
rename, directory fsync, so a reader (or a crash recovery) sees the previous complete document or the new one, never a
torn one. A receipt records what the hardware did (observations, transaction outcome) and is bound to a serving manifest
by its ``metadata.content_digest`` (the manifest's canonical digest, not a hash of the file bytes).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def write_json_atomic(path: Path, document: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if hasattr(os, "O_DIRECTORY"):  # POSIX: persist the rename itself; other platforms have no directory fsync
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_binding(content_digest: str | None) -> dict:
    """The receipt's manifest block: bound to the canonical digest, or explicitly unavailable (never a guessed value)."""
    if content_digest is None:
        return {"status": "unavailable", "content_digest": None, "reason": "manifest_not_produced"}
    if not _DIGEST.match(content_digest):
        raise ValueError(f"not a manifest content digest: {content_digest!r}")
    return {"status": "resolved", "content_digest": content_digest, "reason": None}


@dataclass
class Receipt:
    run_id: str
    manifest: dict = field(default_factory=lambda: manifest_binding(None))
    request: dict = field(default_factory=dict)
    observations: dict = field(default_factory=dict)  #: phase -> probe dict (start, post_apply, pre_restore, stop)
    transaction: dict = field(default_factory=lambda: {"state": "not_requested", "events": []})
    schema_version: int = 1

    def observe(self, phase: str, probe_dict: dict) -> None:
        self.observations[phase] = probe_dict

    def event(self, kind: str, **detail) -> None:
        self.transaction.setdefault("events", []).append({"kind": kind, **detail})

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "run_id": self.run_id, "manifest": self.manifest,
                "request": self.request, "observations": self.observations, "transaction": self.transaction}

    def save(self, path: Path) -> None:
        write_json_atomic(path, self.to_dict())
