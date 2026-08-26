"""Pin-transition marker: the machine declaration that a pin bump is in flight.

RE48/RV78: "a release record exists for the vendor revision" is NOT proof a
bump is in flight -- a manual checkout to any previously recorded revision
satisfies it, and that is the S1 stale-trace incident class. The marker is
the only thing that names a mismatched state as mid-rebase instead of drift.

Lifecycle:
  written  -- bigcherry repin (atomically with the recipes.toml rewrite,
              UNCOMMITTED; step 1 of PIN_BUMP.md commits both together)
  consumed -- pin-status --complete succeeds and the operator commits the
              clear; the marker file is deleted
  stale    -- marker.to_sha no longer equals the current pinned SHA; any
              mismatched state is then drift(stale-marker), never mid-rebase

The marker is a git-tracked file under releases/ precisely so its commit is
the transition declaration: an uncommitted marker is distinguishable from a
committed one, and the two are different states (uncommitted-transition vs
mid-rebase).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import paths


def _default_marker_path() -> Path:
    """Computed at call time (not import time) so a redirected
    bigcherry.core.paths.REPO_ROOT takes effect."""
    return paths.REPO_ROOT / "releases" / "pin-transition.json"


MARKER_PATH = _default_marker_path()
SCHEMA_VERSION = 1


class MarkerError(ValueError):
    pass


@dataclass(frozen=True)
class PinTransition:
    from_sha: str
    to_sha: str
    tag: str
    declaring_commit: str
    set_at: str

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "from_sha": self.from_sha,
            "to_sha": self.to_sha,
            "tag": self.tag,
            "declaring_commit": self.declaring_commit,
            "set_at": self.set_at,
        }


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    json.loads(encoded)  # refuse to publish invalid JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def load(path: Path | None = None) -> PinTransition | None:
    """Read the marker; None when absent. Malformed => MarkerError (fail
    closed: a corrupt marker is a broken state, not an absent one)."""
    path = _default_marker_path() if path is None else path
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarkerError(f"unreadable pin-transition marker: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise MarkerError(f"pin-transition marker has wrong schema_version: {path}")
    for key in ("from_sha", "to_sha", "tag", "declaring_commit", "set_at"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise MarkerError(
                f"pin-transition marker field {key!r} is missing or not a string"
            )
    return PinTransition(
        from_sha=raw["from_sha"],
        to_sha=raw["to_sha"],
        tag=raw["tag"],
        declaring_commit=raw["declaring_commit"],
        set_at=raw["set_at"],
    )


def write(
    from_sha: str,
    to_sha: str,
    tag: str,
    declaring_commit: str,
    path: Path | None = None,
    *,
    now: str | None = None,
) -> PinTransition:
    """Write the marker. Caller (repin) is responsible for the commit."""
    marker = PinTransition(
        from_sha=from_sha,
        to_sha=to_sha,
        tag=tag,
        declaring_commit=declaring_commit,
        set_at=now or datetime.now(timezone.utc).isoformat(),
    )
    _atomic_write_json(
        _default_marker_path() if path is None else path, marker.to_json())
    return marker


def clear(path: Path | None = None) -> bool:
    """Delete the marker. Returns True if a marker was present."""
    path = _default_marker_path() if path is None else path
    if path.is_file():
        path.unlink()
        return True
    return False


def committed_state(path: Path | None = None) -> str:
    """git status of the marker file in its bigcherry repo.

    Returns one of: "absent", "committed-clean", "uncommitted".
    A marker that exists but is not committed is the uncommitted-transition
    state: the transition is declared in the working tree only.

    The repo root is derived from the marker's own location (its parent's
    containing repository), not from paths.REPO_ROOT, so the check is correct
    for any tree the marker may live in -- a marker committed in one repo is
    still uncommitted in another.
    """
    path = _default_marker_path() if path is None else path
    if not path.is_file():
        return "absent"
    import subprocess

    # git status (NOT git diff): an untracked marker -- the exact state
    # immediately after `bigcherry repin`, before `git add` -- is
    # uncommitted, and git diff would silently report it as clean.
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(path.parent),
                "status",
                "--porcelain",
                "--",
                path.name,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip() == "":
            return "committed-clean"
    except OSError:
        pass
    return "uncommitted"
