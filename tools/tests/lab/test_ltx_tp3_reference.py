"""Compile and run the native LTX TP3 oracle; Python is test tooling only.

The lab host mounts /tmp noexec. Build inside BC_WORK_ROOT (or the ignored
repository work directory), and leave no artifacts after the test.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "tools/lab/radiance/ltx-tp3"


def run_checked(args: list[str]) -> None:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False)
    assert result.returncode == 0, result.stdout


def test_native_ltx_tp3_reference() -> None:
    if not shutil.which("cmake") or not (shutil.which("g++") or shutil.which("clang++")):
        pytest.skip("C++20/CMake toolchain unavailable")

    work_root = Path(os.environ.get("BC_WORK_ROOT", str(ROOT / "work")))
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ltx-tp3-test-", dir=work_root) as tmp:
        build = Path(tmp) / "build"
        run_checked(["cmake", "-S", str(REFERENCE), "-B", str(build),
                     "-DBUILD_TESTING=ON"])
        run_checked(["cmake", "--build", str(build), "--parallel", "2"])
        run_checked(["cmake", "--build", str(build), "--target", "test"])
