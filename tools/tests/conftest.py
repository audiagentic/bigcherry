"""Session-wide executable temporary directory for tests.

Some tests compile and execute a binary or load a shared object. On hosts with
noexec system temp mounts, those tests need an exec-capable fallback under the
work root. Each pytest session owns exactly one such directory and cleans it
at unconfigure, including normal test failures. Hard termination can leave an
orphan, which must be reviewed separately rather than blanket deleted.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import sys
import tempfile
import time
import uuid
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from bigcherry.core.context import resolve_work_root
from bigcherry.release.pin_bump import exec_capable_tempdir

# The state lives on pytest's Config, not in a module global, so a nested or
# repeated test run cannot accidentally clean another session's scratch.
_TEMP_STATE = "_bigcherry_exec_temp_state"


def pytest_configure(config) -> None:
    parent = resolve_work_root() / "test-tmp"
    previous_env = os.environ.get("TMPDIR")
    previous_cached = tempfile.tempdir
    directory = exec_capable_tempdir(parent)
    if directory is None:
        return

    # Only the private mkdtemp child returned by our fallback is ours to clean.
    # Reject an unexpected path before directing test fixtures into it.
    info = directory.lstat()
    parent_info = parent.lstat()
    if (directory.parent != parent or not directory.name.startswith("pin-bump-tests-")
            or not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(parent_info.st_mode)
            or info.st_uid != os.getuid() or info.st_mode & 0o777 != 0o700):
        raise RuntimeError(f"unsafe pytest executable temporary directory: {directory}")

    # A hard interruption skips teardown. Record enough provenance to identify
    # an orphan without inferring that it is safe to delete.
    (directory / ".bigcherry-pytest-session.json").write_text(json.dumps({
        "host": socket.gethostname(), "pid": os.getpid(),
        "started_at_unix": time.time(), "purpose": "pytest executable temp fallback",
    }, sort_keys=True) + "\n", encoding="utf-8")
    setattr(config, _TEMP_STATE, (
        directory, info.st_dev, info.st_ino,
        parent_info.st_dev, parent_info.st_ino, previous_env, previous_cached,
    ))
    os.environ["TMPDIR"] = str(directory)
    tempfile.tempdir = str(directory)


@pytest.fixture(autouse=True)
def _restore_process_environment():
    """Undo any change a test (or the CLI entry point it drives) makes to ``os.environ``.

    Campaign entry points export ROCM_PATH/HIP_PATH/PATH for their child processes. Left in place, a later test that
    discovers a toolchain from the environment (hipcc lookup) sees a fixture path from an earlier test."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _no_mounts_under(parent: Path) -> None:
    """Fail closed on bind mounts as well as ordinary nested mounts (Linux)."""
    if not sys.platform.startswith("linux"):
        return
    for row in Path("/proc/self/mountinfo").read_text().splitlines():
        before, separator, _ = row.partition(" - ")
        parts = before.split()
        if not separator or len(parts) < 5:
            raise OSError("invalid mountinfo")
        mount = parts[4].replace("\\040", " ")
        if mount == str(parent) or mount.startswith(str(parent) + "/"):
            raise OSError(f"mount within pytest scratch: {mount}")


def _cleanup_owned(directory: Path, dev: int, inode: int,
                   parent_dev: int, parent_inode: int) -> None:
    """Only remove our original inode, never a replacement at the same path."""
    try:
        parent = directory.parent
        pst = parent.lstat()
        st = directory.lstat()
        if (pst.st_dev, pst.st_ino) != (parent_dev, parent_inode) or not stat.S_ISDIR(pst.st_mode):
            raise OSError("pytest scratch parent was replaced")
        if ((st.st_dev, st.st_ino) != (dev, inode)
                or not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid()):
            raise OSError("pytest scratch directory was replaced or changed owner")
        _no_mounts_under(parent)
        if os.name != "posix" or not shutil.rmtree.avoids_symlink_attacks:
            # Do not attempt potentially unsafe deletion on unsupported hosts.
            raise OSError("safe fd-anchored pytest scratch deletion is unavailable")

        # Quarantine the verified owned tree before destroying its descendants.
        # This prevents a late writer to its original public name from becoming
        # a deletion target. Same-UID/root mutation within the private area is
        # outside this permission boundary, as for build GC.
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            ps = os.fstat(parent_fd)
            if (ps.st_dev, ps.st_ino) != (parent_dev, parent_inode):
                raise OSError("pytest scratch parent changed during open")
            current = os.stat(directory.name, dir_fd=parent_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (dev, inode):
                raise OSError("pytest scratch directory changed during cleanup")
            quarantine = ".pytest-cleanup-" + uuid.uuid4().hex
            os.mkdir(quarantine, mode=0o700, dir_fd=parent_fd)
            try:
                quarantine_fd = os.open(quarantine, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                        dir_fd=parent_fd)
                try:
                    os.rename(directory.name, directory.name,
                              src_dir_fd=parent_fd, dst_dir_fd=quarantine_fd)
                    moved = os.stat(directory.name, dir_fd=quarantine_fd, follow_symlinks=False)
                    if (moved.st_dev, moved.st_ino) != (dev, inode):
                        raise OSError("pytest scratch identity changed after quarantine rename")
                    _no_mounts_under(parent)
                    shutil.rmtree(directory.name, dir_fd=quarantine_fd)
                finally:
                    os.close(quarantine_fd)
            finally:
                # If deletion failed, leave the nonempty private quarantine
                # for human review rather than deleting an unverified tree.
                try:
                    os.rmdir(quarantine, dir_fd=parent_fd)
                except OSError:
                    pass
        finally:
            os.close(parent_fd)
    except FileNotFoundError:
        # The owning test already removed it; never delete a new occupant.
        return
    except OSError as exc:
        warnings.warn(f"pytest scratch cleanup left protected output at {directory}: {exc}",
                      RuntimeWarning, stacklevel=2)


def pytest_unconfigure(config) -> None:
    session = getattr(config, _TEMP_STATE, None)
    if session is None:
        return
    delattr(config, _TEMP_STATE)
    directory, dev, inode, parent_dev, parent_inode, original_env, original_cached = session

    # Preserve a later plugin's deliberate settings; only undo our own values.
    if os.environ.get("TMPDIR") == str(directory):
        if original_env is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = original_env
    if tempfile.tempdir == str(directory):
        tempfile.tempdir = original_cached

    _cleanup_owned(directory, dev, inode, parent_dev, parent_inode)
