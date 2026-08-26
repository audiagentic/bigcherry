"""Effective source tree and source-slice identity."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl  # POSIX only
except ImportError:
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt  # Windows only
except ImportError:
    msvcrt = None  # type: ignore[assignment]


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
    """Compute a tree OID with a temporary index, never the worktree index.

    Only untracked (``??``) files are checked against ``allowed_untracked``
    and rejected if not named there. Modifications to already-tracked
    files (``M``/``D``/etc against upstream's own tree) are never rejected
    here: overlay and patch application both legitimately modify tracked
    upstream files -- that is the entire mechanism of a patch -- and the
    tree OID computed below already captures whatever the true resulting
    state is via ``read-tree``/``add -A``/``write-tree``. Treating every
    such modification as "unexpected" would make this function unable to
    compute an identity for any patched source at all, which defeats its
    purpose: it exists specifically to identify the effective tree AFTER
    overlay and patches, not only a pristine upstream checkout.
    """
    allowed_untracked = allowed_untracked or set()
    status = _run(root, "status", "--porcelain", "--untracked-files=all")
    unexpected: list[str] = []
    for line in status.splitlines():
        if not line:
            continue
        code, _, name = line.partition(" ")
        name = name.strip().replace("\\", "/")
        if code != "??":
            continue
        if name in allowed_untracked:
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


@dataclass(frozen=True)
class SourceAttestation:
    """Immutable identity facts captured for one materialised source tree."""

    upstream_revision: str
    tree_oid: str
    object_format: str
    source_slice_id: str
    allowed_untracked: frozenset[str] = frozenset()


def git_revision(root: Path) -> str:
    """Return the live worktree HEAD revision."""

    return _run(root, "rev-parse", "HEAD")


def git_object_format(root: Path) -> str:
    """Return the live repository object format."""

    return _run(root, "rev-parse", "--show-object-format")


def verify_source_attestation(root: Path, expected: SourceAttestation) -> None:
    """Fail closed unless ``root`` still matches a captured source identity.

    The revision, object format, tree OID, worktree policy, and source-slice
    ID are all re-derived from the live repository.  Values in ``expected``
    are assertions to compare, never a source of truth.
    """

    actual_revision = git_revision(root)
    if actual_revision != expected.upstream_revision:
        raise SourceIdentityError(
            "source HEAD does not match attestation: "
            f"{actual_revision!r} != {expected.upstream_revision!r}"
        )

    actual_format = git_object_format(root)
    if actual_format != expected.object_format:
        raise SourceIdentityError(
            "source object format does not match attestation: "
            f"{actual_format!r} != {expected.object_format!r}"
        )

    allowed_untracked = set(expected.allowed_untracked)
    status = _run(root, "status", "--porcelain", "--untracked-files=all")
    unexpected: list[str] = []
    for line in status.splitlines():
        if not line:
            continue
        code, _, name = line.partition(" ")
        name = name.strip().replace("\\", "/")
        if code == "??" and name not in allowed_untracked:
            unexpected.append(line)

    ignored = _run(root, "status", "--porcelain", "--ignored", "--untracked-files=all")
    ignored_paths = [
        line[3:].strip() for line in ignored.splitlines() if line.startswith("!! ")
    ]
    if unexpected or ignored_paths:
        detail = unexpected + [f"ignored: {path}" for path in ignored_paths]
        raise SourceIdentityError("unexpected source worktree files: " + "; ".join(detail))

    # A clean tree has the HEAD tree exactly; avoid creating a temporary index
    # and re-hashing every file at each worker boundary.  Any tracked change
    # or permitted untracked file takes the existing full-tree path, which
    # preserves exact post-overlay/post-patch identity semantics.
    if not status:
        actual_tree_oid = _run(root, "rev-parse", "HEAD^{tree}")
    else:
        actual_tree_oid = git_tree_oid(root, allowed_untracked=allowed_untracked)
    if actual_tree_oid != expected.tree_oid:
        raise SourceIdentityError(
            "source tree OID does not match attestation: "
            f"{actual_tree_oid!r} != {expected.tree_oid!r}"
        )

    actual_source_slice_id = source_slice_id(
        upstream_revision=actual_revision,
        tree_oid=actual_tree_oid,
        object_format=actual_format,
    )
    if actual_source_slice_id != expected.source_slice_id:
        raise SourceIdentityError(
            "source_slice_id does not match live source facts: "
            f"{actual_source_slice_id!r} != {expected.source_slice_id!r}"
        )


def atomic_write_json(
    path: Path, document: object, *, fsync_file: bool = True, fsync_parent: bool = True,
    read_only: bool = False,
) -> None:
    """Write JSON to ``path`` via temp-file + fsync + ``os.replace()``.

    PA12 (source/patch identity hardening L6.1): HI82's manifest writer
    (patch/source.py::_write_manifest) already did this; canonical campaign
    metadata (campaign/build.py::materialize_source()) used a plain
    ``write_text()``, so a crash mid-write could leave truncated metadata
    beside an otherwise-valid source worktree. Shared here so both call
    sites use one implementation instead of maintaining two.

    ``fsync_parent`` additionally fsyncs the containing directory after the
    rename -- on POSIX, the rename itself is not guaranteed durable until
    the directory entry is flushed; skipped automatically where the
    platform does not support directory fsync (e.g. Windows).

    ``read_only`` (adversarial-review follow-up): chmods the final file to
    0o444 after the atomic replace. A manifest whose own fields are later
    compared against re-derived live facts (HI82's ``_verify_reuse()``) is
    only as trustworthy as those fields being un-editable after the one
    legitimate writer produced them -- an in-place edit of the manifest
    JSON is a much easier attack than mutating a git worktree convincingly.
    This is NOT a defense against a privileged/determined attacker who can
    chmod the file back to writable first (that requires an independently
    authoritative binding, which this does not implement); it is real
    defense-in-depth against accidental, buggy, or casual mutation, which
    is this review's primary threat model throughout.
    """
    path = Path(path)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            if fsync_file:
                os.fsync(handle.fileno())
        if read_only:
            os.chmod(temporary_path, 0o444)
        os.replace(temporary_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary_path.chmod(0o644)
        temporary_path.unlink(missing_ok=True)
        raise

    if fsync_parent:
        try:
            parent_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            # Windows (and some filesystems) do not support opening a
            # directory for fsync -- the temp-file-plus-replace already
            # gives atomicity of content, this is best-effort durability.
            return
        try:
            os.fsync(parent_fd)
        except OSError:
            pass
        finally:
            os.close(parent_fd)


class PlanLockTimeout(RuntimeError):
    """A materialization-plan lock could not be acquired within the deadline."""


def _try_lock(fd: int) -> bool:
    """One non-blocking attempt to take an exclusive OS advisory lock on
    ``fd``. True on success. The lock is tied to the open file descriptor:
    the OS releases it automatically if this process dies (crash, kill -9,
    power loss) without an explicit unlock -- this is what gives PA12's
    locking its "auto-release on process exit" property for free, rather
    than needing a separate liveness/heartbeat mechanism."""
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False
    if msvcrt is not None:
        try:
            # msvcrt has no whole-file lock; lock one byte at a fixed offset
            # as the mutex region -- every locker of the same path contends
            # on the same byte.
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    raise RuntimeError("no supported file-locking primitive on this platform")


def _unlock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        with contextlib.suppress(OSError):
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def plan_lock(work_root: Path, plan_id: str, *, timeout_seconds: float = 300.0, poll_interval: float = 0.2):
    """Exclusive critical section for one materialization plan (PA12 L6.2).

    Two processes computing the same materialization plan concurrently must
    not both create/mutate the same worktree/cache record -- this serializes
    the whole inspect-cache -> materialize -> attest -> publish-metadata
    sequence per ``plan_id``. The lock file lives under
    ``<work_root>/locks/source-<plan_id>.lock``, matching the review's
    suggested layout; callers supply ``work_root`` explicitly rather than
    this function guessing it via a fixed ``Path.parents[N]`` offset.

    Process-safe (real OS advisory lock, not a lockfile-existence convention
    -- immune to a stale lock file surviving a crash) and auto-released on
    process exit (the OS drops the lock when the fd closes, including an
    abnormal process exit). Raises ``PlanLockTimeout`` rather than blocking
    forever if the lock cannot be acquired within ``timeout_seconds``.
    """
    lock_dir = Path(work_root) / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"source-{plan_id}.lock"

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        deadline = time.monotonic() + timeout_seconds
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise PlanLockTimeout(
                    f"could not acquire materialization-plan lock for "
                    f"{plan_id!r} at {lock_path} within {timeout_seconds}s "
                    f"-- another process is likely materializing this same plan"
                )
            time.sleep(poll_interval)
        try:
            yield lock_path
        finally:
            _unlock(fd)
    finally:
        os.close(fd)
