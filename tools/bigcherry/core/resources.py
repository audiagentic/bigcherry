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
        # RD100: a lock this instance never successfully acquired must never
        # be released by it -- release() used to be callable unconditionally,
        # so a caller that iterated every lock it *tried* to acquire (not
        # just the ones that succeeded) could delete another process's live
        # lock out from under it, breaking mutual exclusion entirely (two
        # processes then both entered the same build directory concurrently).
        self._acquired = False

    def acquire(self, *, timeout_seconds: float = 300.0, poll_interval: float = 0.2) -> None:
        """Claim the resource, waiting for a live contender to finish first.

        RD100 (gpt-auto-agent review follow-up): this used to be fail-fast --
        a single ``mkdir()`` attempt, immediate ``ResourceError`` on
        contention. On the documented shared multi-agent Brutus host, two
        legitimate, non-conflicting build requests contending for the same
        resource (e.g. two lanes both wanting a GPU claim, or the same
        build-plan resource from two processes racing to publish the same
        content-addressed result) is a real, expected occurrence, not a
        hypothetical -- fail-fast there just means the second one dies for
        no reason instead of proceeding once the first is done. Bounded
        (not indefinite) so a genuinely stuck/abandoned lock still surfaces
        as an error rather than hanging forever.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                self.path.mkdir()
                break
            except FileExistsError as exc:
                if time.monotonic() >= deadline:
                    raise ResourceError(
                        f"resource is already claimed: {self.resource_id} "
                        f"(waited {timeout_seconds}s)"
                    ) from exc
                time.sleep(poll_interval)
        owner = {
            "hostname": socket.gethostname(), "pid": os.getpid(),
            "started_at": time.time(), "resource_id": self.resource_id,
        }
        try:
            (self.path / "owner.json").write_text(
                json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError:
            # The directory claim succeeded but recording ownership didn't --
            # leaving the bare directory behind would strand the lock
            # forever (release() below refuses to touch a lock this instance
            # never marked _acquired, and no OTHER instance can tell this
            # abandoned directory apart from a live one via inspect() alone).
            self.path.rmdir()
            raise
        self._acquired = True

    def release(self) -> None:
        if not self._acquired:
            # Never held (or already released) by this instance -- a no-op,
            # not a delete of whatever/whoever currently owns the path.
            return
        owner = self.path / "owner.json"
        owner.unlink(missing_ok=True)
        try:
            self.path.rmdir()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ResourceError(f"resource lock is not empty: {self.path}") from exc
        self._acquired = False

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
