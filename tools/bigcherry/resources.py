"""Host-local atomic resource claims."""

from __future__ import annotations

import json
import os
import re
import socket
import time
from pathlib import Path


class ResourceError(RuntimeError):
    pass


class ResourceLock:
    def __init__(self, root: Path, resource_id: str):
        self.root = root.resolve()
        self.resource_id = resource_id
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", resource_id)
        self.path = self.root / (safe + ".lock")

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError as exc:
            raise ResourceError(f"resource is already claimed: {self.resource_id}") from exc
        owner = {
            "hostname": socket.gethostname(), "pid": os.getpid(),
            "started_at": time.time(), "resource_id": self.resource_id,
        }
        (self.path / "owner.json").write_text(json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8")

    def release(self) -> None:
        owner = self.path / "owner.json"
        owner.unlink(missing_ok=True)
        try:
            self.path.rmdir()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ResourceError(f"resource lock is not empty: {self.path}") from exc

    def inspect(self) -> dict[str, object] | None:
        if not self.path.is_dir():
            return None
        owner = self.path / "owner.json"
        return json.loads(owner.read_text(encoding="utf-8")) if owner.is_file() else {}

    def break_explicit(self) -> None:
        """Explicit operator action; never called automatically by acquire."""
        info = self.inspect()
        if info is None:
            return
        owner = self.path / "owner.json"
        owner.unlink(missing_ok=True)
        self.path.rmdir()
