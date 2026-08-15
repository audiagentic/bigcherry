"""Effective source tree and source-slice identity."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


class SourceIdentityError(ValueError):
    pass


def _run(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise SourceIdentityError(detail.strip()) from exc
    return result.stdout.strip()


def git_tree_oid(root: Path, *, allowed_untracked: set[str] | None = None) -> str:
    """Compute a tree OID with a temporary index, never the worktree index."""
    allowed_untracked = allowed_untracked or set()
    status = _run(root, "status", "--porcelain", "--untracked-files=all")
    unexpected: list[str] = []
    for line in status.splitlines():
        if not line:
            continue
        code, _, name = line.partition(" ")
        name = name.strip().replace("\\", "/")
        if code == "??" and name in allowed_untracked:
            continue
        unexpected.append(line)
    ignored = _run(root, "status", "--porcelain", "--ignored", "--untracked-files=all")
    ignored_paths = [
        line[3:].strip() for line in ignored.splitlines() if line.startswith("!! ")
    ]
    if unexpected or ignored_paths:
        detail = unexpected + [f"ignored: {path}" for path in ignored_paths]
        raise SourceIdentityError("unexpected source worktree files: " + "; ".join(detail))

    with tempfile.NamedTemporaryFile(prefix="bigcherry-index-", delete=False) as handle:
        index = Path(handle.name)
    try:
        index.unlink(missing_ok=True)
        child_env = os.environ.copy()
        child_env["GIT_INDEX_FILE"] = str(index)
        _run(root, "read-tree", "HEAD", env=child_env)
        _run(root, "add", "-A", env=child_env)
        return _run(root, "write-tree", env=child_env)
    finally:
        index.unlink(missing_ok=True)


def source_slice_id(*, upstream_revision: str, tree_oid: str, object_format: str = "sha1") -> str:
    payload = {
        "source_identity_schema": 1,
        "upstream_revision": upstream_revision,
        "git_object_format": object_format,
        "source_tree_oid": tree_oid,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(b"bigcherry/source-slice/v1\0" + encoded, digest_size=16).hexdigest()


def describe(*, root: Path, upstream_revision: str, allowed_untracked: set[str] | None = None) -> dict[str, object]:
    tree_oid = git_tree_oid(root, allowed_untracked=allowed_untracked)
    return {
        "source_identity_schema": 1,
        "upstream_revision": upstream_revision,
        "git_object_format": _run(root, "rev-parse", "--show-object-format"),
        "source_tree_oid": tree_oid,
        "source_slice_id": source_slice_id(
            upstream_revision=upstream_revision,
            tree_oid=tree_oid,
            object_format=_run(root, "rev-parse", "--show-object-format"),
        ),
    }
