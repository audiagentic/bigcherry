"""Cross-platform host-local kernel file locks used by the jobs service.

Unlike the legacy mkdir ResourceLock, ownership is released by the OS when a
process dies.  This primitive is for serialization/local-executor fencing; it
is never a distributed lock and must only protect state on a local filesystem.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


class HostLockError(RuntimeError):
    pass


class HostFileLock:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._handle = None

    def acquire(self, *, timeout_seconds: float = 300.0, poll_interval: float = 0.05) -> None:
        if self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                _lock(handle)
                self._handle = handle
                return
            except (BlockingIOError, OSError) as exc:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise HostLockError(f"timed out locking {self.path}") from exc
                time.sleep(poll_interval)

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            _unlock(handle)
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> "HostFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _lock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        # msvcrt.locking() locks bytes from the current file position. Ensure
        # a single lock byte exists without appending a byte on every acquire.
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
