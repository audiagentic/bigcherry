"""Session-wide test temp directory that can execute what tests build in it.

Some tests compile a small program or shared object into a temporary directory and run or load it (the plugin ABI reader,
the rad0013 patched callback). Where the system temp directory is mounted noexec (a hardened container's /tmp) those tests
fail with "failed to map segment" / "permission denied" for reasons unrelated to the code under test. pin-bump already
guards its own patch-test runs this way (release.pin_bump.exec_capable_tempdir); this applies the same probe once for the
whole session: where the system temp directory executes (CI) nothing changes, otherwise TMPDIR and Python's tempfile
default move under the work root ($BC_WORK_ROOT, exec-capable by construction).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.release.pin_bump import exec_capable_tempdir  # noqa: E402


def pytest_configure(config) -> None:  # noqa: ARG001 - pytest hook signature
    work_root = os.environ.get("BC_WORK_ROOT")
    if not work_root:
        return
    directory = exec_capable_tempdir(Path(work_root) / "test-tmp")
    if directory is None:
        return
    os.environ["TMPDIR"] = str(directory)
    tempfile.tempdir = str(directory)
