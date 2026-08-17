"""HI68: compile and run the host-side canary decision unit test.

The transition logic under test lives in a GPU-free header
(src/ggml/src/ggml-cuda/hip-autotune-canary.h -- see the path constant
below) so it can be exercised by ANY C++17 host compiler without a device,
driver, or ggml build. This test locates a compiler (ROCm clang++ first,
then anything on PATH that calls itself clang++ or cl), builds
tools/tests/canary_decision_host_test.cpp against the header, runs it, and
requires the OK marker on stdout.

It skips rather than fails when no compiler exists: the contract is that
the test RAN, not that a machine without any C++ toolchain still passes CI.
"""

import os
import platform
import shutil
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CANARY_HEADER = os.path.join(
    REPO_ROOT, "src", "ggml", "src", "ggml-cuda", "hip-autotune-canary.h"
)
HOST_TEST_CPP = os.path.join(REPO_ROOT, "tools", "tests", "canary_decision_host_test.cpp")
OK_MARKER = "CANARY_DECISION_HOST_TEST_OK"


def _find_compiler():
    """Return (compiler, extra-args) for a C++17 host compiler, or None."""
    candidates = []
    if platform.system() == "Windows":
        rocm = os.environ.get("HIP_PATH") or r"C:\Program Files\AMD\ROCm\7.1"
        candidates.append(os.path.join(rocm, "bin", "clang++.exe"))
    for name in ("clang++", "cl"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for candidate in candidates:
        if os.path.exists(candidate):
            extra = []
            if os.path.basename(candidate).lower() == "cl.exe":
                extra = ["/EHsc", "/std:c++17"]
            return candidate, extra
    return None


class TestHi68CanaryDecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        found = _find_compiler()
        if found is None:
            raise unittest.SkipTest(
                "no C++17 host compiler found (tried ROCm clang++, clang++, cl)")
        cls.compiler, cls.extra_args = found

    def test_canary_decision_host_test(self):
        self.assertTrue(os.path.isfile(CANARY_HEADER), CANARY_HEADER)
        self.assertTrue(os.path.isfile(HOST_TEST_CPP), HOST_TEST_CPP)
        workdir = tempfile.mkdtemp(prefix="hi68-canary-")
        try:
            exe = os.path.join(workdir, "canary_decision_host_test.exe"
                               if platform.system() == "Windows" else "canary_decision_host_test")
            cmd = [self.compiler] + self.extra_args + [
                "-I", os.path.dirname(CANARY_HEADER),
                HOST_TEST_CPP, "-o", exe,
            ]
            if not any(a.startswith("/std:") for a in self.extra_args) and \
                    os.path.basename(self.compiler).lower().startswith("clang"):
                cmd.insert(1, "-std=c++17")
            build = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if build.returncode != 0:
                self.fail(
                    f"host canary test failed to compile with {self.compiler}:\n"
                    f"{build.stderr[-4000:]}")
            run = subprocess.run([exe], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn(OK_MARKER, run.stdout)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
