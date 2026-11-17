"""Read-only legacy run scanner and opt-in, no-overwrite manifest adoption (WSL09).

Only flat runs/<run-name>/ directories are adoptable. No existing bulk is moved
or deleted. Scan all regular files (including binary), reject symlinks, special
files, unreadable files, device changes and concurrent tree mutations. A build
edge is recorded as source-slice/build-plan, not just a plan-id, so build GC
protects precisely the referenced build.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path

from . import run_manifest as rm

CHUNK_SIZE = 65536
SCANNER_VERSION = "legacy-build-path-v1"
# Component boundaries, not substring/substring-of-hex matches. A relative
# builds/... reference and an absolute <work-root>/builds/... match identically.
_BUILD = re.compile(rb"(?<![A-Za-z0-9_.-])builds/([0-9a-f]{32})/([0-9a-f]{32})(?![A-Za-z0-9_.-])")
_OVERLAP = len(b"builds/") + 32 + 1 + 32 + 1  # full match plus following boundary
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


class LegacyRunError(RuntimeError):
    """An incomplete scan must never produce a manifest or clear a GC blocker."""


def _identity(st: os.stat_result) -> tuple[int, int]:
    return st.st_dev, st.st_ino


def _same_snapshot(before: os.stat_result, after: os.stat_result) -> bool:
    return (_identity(before) == _identity(after)
            and before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
            and before.st_ctime_ns == after.st_ctime_ns)


def _signature(st: os.stat_result) -> tuple[int, int, int, int, int]:
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _tree_stamp(snapshots: dict[str, tuple[int, int, int, int, int]]) -> dict:
    files = [value for name, value in snapshots.items() if name.startswith("f:")]
    digest = hashlib.sha256()
    for name, value in sorted(snapshots.items()):
        # Ignore directory mtime/ctime/size: publication creates a manifest
        # and removes the lock, changing those fields after the scan.
        stable = value if name.startswith("f:") else value[:2]
        digest.update(repr((name, stable)).encode("utf-8") + b"\n")
    return {"file_count": len(files),
            "total_bytes": sum(value[2] for value in files),
            "max_mtime_ns": max((value[3] for value in files), default=0),
            "directory_count": len(snapshots) - len(files),
            "metadata_digest": digest.hexdigest()}


def _stat_tree(fd: int, device: int, *, prefix: str = "", top: bool = True,
               skip_lock: bool = True, skip_manifest: bool = False) -> dict:
    """Stat every child without following links; used at publication and GC time.

    The snapshot includes file identity, size, mtime_ns and ctime_ns and
    directory identities/membership. An unreadable or special item fails closed.
    """
    before = os.fstat(fd)
    if before.st_dev != device:
        raise LegacyRunError("legacy run tree crosses device boundary")
    snapshots = {}
    with os.scandir(fd) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        if top and ((skip_lock and name == rm.MANIFEST_NAME + ".lock")
                    or (skip_manifest and name == rm.MANIFEST_NAME)):
            continue
        st = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if st.st_dev != device or stat.S_ISLNK(st.st_mode):
            raise LegacyRunError(f"symlink or device change during tree verification: {name}")
        child = f"{prefix}/{name}" if prefix else name
        if stat.S_ISDIR(st.st_mode):
            sub = os.open(name, _DIR_FLAGS, dir_fd=fd)
            try:
                if _identity(os.fstat(sub)) != _identity(st):
                    raise LegacyRunError(f"directory replaced during tree verification: {child}")
                snapshots.update(_stat_tree(sub, device, prefix=child, top=False))
            finally:
                os.close(sub)
        elif stat.S_ISREG(st.st_mode):
            if not (st.st_mode & 0o444):
                raise LegacyRunError(f"unreadable file during tree verification: {child}")
            snapshots[f"f:{child}"] = _signature(st)
        else:
            raise LegacyRunError(f"special file during tree verification: {child}")
    if not _same_snapshot(before, os.fstat(fd)):
        raise LegacyRunError(f"directory changed during tree verification: {prefix or '.'}")
    snapshots[f"d:{prefix}"] = _signature(before)
    return snapshots


def _metadata() -> dict:
    return {"scanner_version": SCANNER_VERSION, "files_scanned": 0,
            "bytes_scanned": 0, "min_mtime": None, "max_mtime": None}


def _record(meta: dict, st: os.stat_result, bytes_read: int) -> None:
    meta["files_scanned"] += 1
    meta["bytes_scanned"] += bytes_read
    meta["min_mtime"] = min(meta["min_mtime"], st.st_mtime) if meta["min_mtime"] is not None else st.st_mtime
    meta["max_mtime"] = max(meta["max_mtime"], st.st_mtime) if meta["max_mtime"] is not None else st.st_mtime


def _scan_file(fd: int, name: str, expected: os.stat_result, device: int,
               refs: set[str], meta: dict) -> None:
    if not stat.S_ISREG(expected.st_mode) or expected.st_dev != device:
        raise LegacyRunError(f"untrusted special/cross-device file {name}")
    # Also refuse mode-000 on privileged invocations: root being able to open
    # a file does not make an otherwise unreadable legacy scan complete.
    if not (expected.st_mode & 0o444):
        raise LegacyRunError(f"unreadable file {name}: no read permission bits")
    f = os.open(name, _FILE_FLAGS, dir_fd=fd)
    try:
        opened = os.fstat(f)
        if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(expected):
            raise LegacyRunError(f"file changed or is not regular: {name}")
        pending = b""
        # After trimming, byte zero is a retained boundary already examined.
        # Never reconsider it as a fresh match at the beginning of a buffer.
        examined_first = False
        count = 0
        while True:
            chunk = os.read(f, CHUNK_SIZE)
            if not chunk:
                break
            count += len(chunk)
            pending += chunk
            # Reserve a full match-width of lookahead. A path ending exactly
            # at a chunk edge may be followed by more hex/slug characters;
            # matching it prematurely would create a false build pin.
            limit = max(0, len(pending) - _OVERLAP)
            if limit:
                for hit in _BUILD.finditer(pending):
                    if hit.start() < limit and (hit.start() != 0 or not examined_first):
                        refs.add(hit.group(1).decode("ascii") + "/" + hit.group(2).decode("ascii"))
                # Leave the byte immediately before the unprocessed suffix:
                # the regex's negative lookbehind must see the real boundary.
                pending = pending[limit - 1:]
                examined_first = True
        for hit in _BUILD.finditer(pending):
            if hit.start() != 0 or not examined_first:
                refs.add(hit.group(1).decode("ascii") + "/" + hit.group(2).decode("ascii"))
        if not _same_snapshot(opened, os.fstat(f)):
            raise LegacyRunError(f"file changed during scan: {name}")
        _record(meta, opened, count)
    finally:
        os.close(f)


def _scan_dir(fd: int, device: int, refs: set[str], meta: dict, *,
              top: bool = False, skip_lock: bool = False,
              snapshots: dict | None = None, prefix: str = "") -> None:
    before = os.fstat(fd)
    try:
        with os.scandir(fd) as entries:
            names = sorted(e.name for e in entries)
    except OSError as exc:
        raise LegacyRunError(f"cannot list run directory: {exc}") from exc
    for name in names:
        if top and skip_lock and name == rm.MANIFEST_NAME + ".lock":
            # Only the transient lock acquired by this apply invocation;
            # dry runs scan any preexisting lock file as historical content.
            continue
        if name in {".", ".."}:
            continue
        try:
            st = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if st.st_dev != device or stat.S_ISLNK(st.st_mode):
                raise LegacyRunError(f"symlink or device change: {name}")
            child = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(st.st_mode):
                sub = os.open(name, _DIR_FLAGS, dir_fd=fd)
                try:
                    observed = os.fstat(sub)
                    if _identity(observed) != _identity(st):
                        raise LegacyRunError(f"directory replaced during scan: {name}")
                    _scan_dir(sub, device, refs, meta, snapshots=snapshots, prefix=child)
                    if not _same_snapshot(observed, os.fstat(sub)):
                        raise LegacyRunError(f"directory changed during scan: {name}")
                finally:
                    os.close(sub)
            elif stat.S_ISREG(st.st_mode):
                _scan_file(fd, name, st, device, refs, meta)
                if snapshots is not None:
                    snapshots[f"f:{child}"] = _signature(st)
            else:
                raise LegacyRunError(f"special file: {name}")
        except OSError as exc:
            raise LegacyRunError(f"cannot fully scan {name}: {exc}") from exc
    # Directory size and times detect unobserved additions/removals while
    # scanning; the only permitted write during an apply scan is our lock.
    if not _same_snapshot(before, os.fstat(fd)):
        raise LegacyRunError("run directory changed during scan")
    if snapshots is not None:
        snapshots[f"d:{prefix}"] = _signature(before)


def _public_evidence(meta: dict) -> dict:
    return {**meta, "min_mtime": rm.utc_stamp(meta["min_mtime"]) if meta["min_mtime"] is not None else None,
            "max_mtime": rm.utc_stamp(meta["max_mtime"]) if meta["max_mtime"] is not None else None}


def _typed(refs: set[str]) -> list[dict]:
    return [{"type": "build", "id": value, "role": "legacy-scanned"} for value in sorted(refs)]


def _open_runs(root: Path) -> tuple[int, int]:
    # Anchor all descendant operations at a real, unsymlinked work root.
    root_fd = os.open(root, _DIR_FLAGS)
    try:
        runs_fd = os.open("runs", _DIR_FLAGS, dir_fd=root_fd)
        if os.fstat(runs_fd).st_dev != os.fstat(root_fd).st_dev:
            raise LegacyRunError("runs/ crosses filesystem boundary")
        return root_fd, runs_fd
    except BaseException:
        os.close(root_fd)
        raise


def scan_loose_file(root: Path, name: str) -> tuple[list[dict], dict]:
    """Fully scan a flat *regular file* in runs/ without writing to it.

    Build GC can classify text/binary evidence this way; unscannable entries
    remain global blockers. The directory inventory still reports such files
    as outside the managed layout and never permits run cleanup on them.
    """
    root_fd, runs_fd = _open_runs(Path(root))
    try:
        st = os.stat(name, dir_fd=runs_fd, follow_symlinks=False)
        refs: set[str] = set()
        meta = _metadata()
        _scan_file(runs_fd, name, st, os.fstat(runs_fd).st_dev, refs, meta)
        return _typed(refs), _public_evidence(meta)
    except OSError as exc:
        raise LegacyRunError(f"cannot scan loose run file {name}: {exc}") from exc
    finally:
        os.close(runs_fd)
        os.close(root_fd)


def _reject_run_mounts(root: Path, name: str) -> None:
    """Refuse mounts at runs/, on a flat run, or anywhere inside it.

    This also detects same-device bind mounts: comparing st_dev alone cannot
    prevent writing a manifest into a bind-mounted external directory.
    """
    from . import build_gc

    runs = root / "runs"
    run = runs / name
    try:
        mounts = build_gc._mount_points()
    except build_gc.BuildGCError as exc:
        raise LegacyRunError(f"cannot verify legacy run mounts: {exc}") from exc
    for mount in mounts:
        if mount.is_relative_to(runs) and (run.is_relative_to(mount) or mount.is_relative_to(run)):
            raise LegacyRunError(f"mount point on legacy run path: {mount}")


def verify_adoption(root: Path, name: str, manifest: dict) -> None:
    """A flat manifest is authoritative only with scanner provenance and a live tree stamp.

    This is metadata validation, not a cryptographic tamper seal: the historical
    work root is operated by a single trusted user.
    """
    scan = manifest.get("legacy_scan")
    if (manifest.get("provenance") != "legacy_inferred"
            or not isinstance(scan, dict)
            or scan.get("scanner_version") != SCANNER_VERSION):
        raise LegacyRunError("flat run lacks trusted legacy scanner provenance")
    stamp = scan.get("tree_stamp")
    fields = ("file_count", "total_bytes", "max_mtime_ns", "directory_count")
    if (not isinstance(stamp, dict)
            or any(type(stamp.get(field)) is not int or stamp[field] < 0 for field in fields)
            or type(scan.get("files_scanned")) is not int
            or type(scan.get("bytes_scanned")) is not int
            or scan["files_scanned"] != stamp["file_count"]
            or scan["bytes_scanned"] != stamp["total_bytes"]
            or not isinstance(stamp.get("metadata_digest"), str)
            or re.fullmatch(r"[0-9a-f]{64}", stamp["metadata_digest"]) is None):
        raise LegacyRunError("flat run lacks a valid legacy tree stamp")

    _reject_run_mounts(root, name)
    root_fd, runs_fd = _open_runs(root)
    try:
        st = os.stat(name, dir_fd=runs_fd, follow_symlinks=False)
        if not stat.S_ISDIR(st.st_mode):
            raise LegacyRunError("legacy run is not a real directory")
        run_fd = os.open(name, _DIR_FLAGS, dir_fd=runs_fd)
        try:
            if _identity(os.fstat(run_fd)) != _identity(st):
                raise LegacyRunError("legacy run directory replaced")
            current = _tree_stamp(_stat_tree(run_fd, st.st_dev, skip_manifest=True))
            if current != stamp:
                raise LegacyRunError("legacy run tree stamp changed since adoption")
        finally:
            os.close(run_fd)
    finally:
        os.close(runs_fd)
        os.close(root_fd)


def _manifest_exists(fd: int) -> bool:
    try:
        st = os.stat(rm.MANIFEST_NAME, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(st.st_mode):
        raise LegacyRunError("existing run manifest is a symlink")
    return True


def _publish(fd: int, manifest: dict) -> None:
    """Link fully fsynced temp into place; link(2) fails if destination exists.

    Unlike os.replace, this cannot overwrite a manifest created concurrently.
    All writes are relative to the verified run directory descriptor.
    """
    raw = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")
    tmp = rm.MANIFEST_NAME + ".tmp-" + uuid.uuid4().hex
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    f = os.open(tmp, flags, 0o600, dir_fd=fd)
    try:
        os.fchmod(f, 0o644)
        view = memoryview(raw)
        while view:
            n = os.write(f, view)
            if not n:
                raise LegacyRunError("short write of run manifest")
            view = view[n:]
        os.fsync(f)
        os.link(tmp, rm.MANIFEST_NAME, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        os.fsync(fd)
    finally:
        os.close(f)
        os.unlink(tmp, dir_fd=fd)


def _one(root: Path, runs_fd: int, name: str, apply: bool) -> dict:
    row: dict = {"run_id": name, "path": str(root / "runs" / name),
                 "status": "error", "references": []}
    try:
        _reject_run_mounts(root, name)
        st = os.stat(name, dir_fd=runs_fd, follow_symlinks=False)
        if not stat.S_ISDIR(st.st_mode):
            raise LegacyRunError("not a real run directory")
        run_fd = os.open(name, _DIR_FLAGS, dir_fd=runs_fd)
        try:
            initial = os.fstat(run_fd)
            if _identity(initial) != _identity(st) or initial.st_dev != os.fstat(runs_fd).st_dev:
                raise LegacyRunError("legacy run directory changed device/identity")

            def scan_and_publish() -> None:
                if _manifest_exists(run_fd):
                    row["status"] = "existing"
                    return
                refs: set[str] = set()
                evidence = _metadata()
                scanned: dict = {}
                _scan_dir(run_fd, initial.st_dev, refs, evidence, top=True,
                          skip_lock=apply, snapshots=scanned)
                evidence["tree_stamp"] = _tree_stamp(scanned)
                # A renamed/moved run cannot acquire a manifest at the
                # original name. The lock and the writer are fd-anchored too.
                observed = os.stat(name, dir_fd=runs_fd, follow_symlinks=False)
                if _identity(observed) != _identity(initial):
                    raise LegacyRunError("legacy run directory moved or replaced")
                row["references"] = _typed(refs)
                row["scan"] = _public_evidence(evidence)
                row["status"] = "would_write"
                if apply:
                    # A completed read is not enough: earlier files may have
                    # changed in place without modifying their parent directory.
                    verified = _stat_tree(run_fd, initial.st_dev)
                    if verified != scanned or _tree_stamp(verified) != evidence["tree_stamp"]:
                        raise LegacyRunError("legacy run tree changed after scan")
                    _reject_run_mounts(root, name)  # recheck immediately before writing
                    m = rm.new_manifest(run_id=name, kind="legacy",
                                        purpose="Legacy run inventory; original provenance unknown",
                                        plan_item=None, engine="unknown", argv=[],
                                        owner={"uid": initial.st_uid}, references=row["references"],
                                        now=evidence["min_mtime"] if evidence["min_mtime"] is not None else initial.st_mtime)
                    m["provenance"] = "legacy_inferred"
                    m["legacy_scan"] = row["scan"]
                    rm.validate_manifest(m)
                    _publish(run_fd, m)
                    # Publication is fd-anchored, so a renamed directory can
                    # receive a manifest outside runs/. Detect and roll it back.
                    try:
                        current = os.stat(name, dir_fd=runs_fd, follow_symlinks=False)
                        if (not stat.S_ISDIR(current.st_mode)
                                or _identity(current) != _identity(os.fstat(run_fd))):
                            raise LegacyRunError("run moved during publication")
                    except (OSError, LegacyRunError) as exc:
                        try:
                            os.unlink(rm.MANIFEST_NAME, dir_fd=run_fd)
                            os.fsync(run_fd)
                        except OSError as rollback:
                            raise LegacyRunError(f"run moved; manifest rollback failed: {rollback}") from rollback
                        raise LegacyRunError("run moved during publication; manifest rolled back") from exc
                    row["status"] = "written"

            if apply:
                # The existing cooperative run-manifest lock, but opened and
                # removed by descriptor so a swapped path is never followed.
                with rm._Lock(root / "runs" / name, dir_fd=run_fd):
                    scan_and_publish()
            else:
                scan_and_publish()  # No lock or other writes in a dry run.
        finally:
            os.close(run_fd)
    except (OSError, LegacyRunError, rm.ManifestError) as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def migrate(root: Path, *, apply: bool = False) -> dict:
    """Scan all legacy flat run directories, optionally add v1 manifests.

    This never moves/deletes run files. Every unscannable path is a reported
    error and remains a GC blocker; all writes are opt-in. Output is detailed
    per run so an operator can inspect before invoking --apply.
    """
    root = Path(root)
    result: dict = {"root": str(root), "read_only": not apply, "runs": [], "loose_files": [],
                    "totals": {"directories": 0, "would_write": 0, "written": 0,
                               "existing": 0, "errors": 0, "loose_files": 0}}
    try:
        root_fd, runs_fd = _open_runs(root)
    except (OSError, LegacyRunError) as exc:
        result["totals"]["errors"] = 1
        result["error"] = f"untrusted runs/ root: {exc}"
        return result
    try:
        with os.scandir(runs_fd) as entries:
            names = sorted(e.name for e in entries)
        for name in names:
            try:
                st = os.stat(name, dir_fd=runs_fd, follow_symlinks=False)
                if stat.S_ISDIR(st.st_mode) and name in rm.NESTED_RUN_KINDS:
                    continue  # already managed kind directory, not a flat run
                if stat.S_ISDIR(st.st_mode):
                    result["totals"]["directories"] += 1
                    row = _one(root, runs_fd, name, apply)
                    result["runs"].append(row)
                    result["totals"]["errors" if row["status"] == "error" else row["status"]] += 1
                elif stat.S_ISREG(st.st_mode):
                    result["totals"]["loose_files"] += 1
                    row = {"path": str(root / "runs" / name)}
                    try:
                        refs, evidence = scan_loose_file(root, name)
                        row.update(status="scanned", references=refs, scan=evidence)
                    except (LegacyRunError, OSError) as exc:
                        row.update(status="error", error=str(exc))
                        result["totals"]["errors"] += 1
                    result["loose_files"].append(row)
                else:
                    result["runs"].append({"path": str(root / "runs" / name), "run_id": name,
                                           "status": "error", "references": [],
                                           "error": "untrusted symlink or special run entry"})
                    result["totals"]["errors"] += 1
            except OSError as exc:
                result["runs"].append({"path": str(root / "runs" / name), "run_id": name,
                                       "status": "error", "references": [], "error": str(exc)})
                result["totals"]["errors"] += 1
    except OSError as exc:
        result["totals"]["errors"] += 1
        result["error"] = f"incomplete runs/ enumeration: {exc}"
    finally:
        os.close(runs_fd)
        os.close(root_fd)
    return result
