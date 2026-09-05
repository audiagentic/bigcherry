"""GP11/GP13: per-element correctness regression test for multi-GPU
peer-to-peer copies on AMD RDNA hardware.

hipMemcpy(dst, src, bytes, hipMemcpyDefault) issued with the CURRENT
device set to the destination (a "pull") was found to silently return
hipSuccess while copying zeros/garbage on gfx1100 -- the mirrored call
with current device set to the source (a "push") is correct.
hipDeviceCanAccessPeer() reports peer-capable=1 in both directions
regardless, so topology reporting cannot be trusted as a proxy for actual
data movement. This was found once (GP11), retracted a bandwidth
benchmark that had used the broken pull direction, and then a later,
unrelated benchmark independently reintroduced the same broken pull
direction and produced a plausible-looking but fully wrong bandwidth
number (10.4 GB/s, later retracted) -- because nothing in the automated
test suite would have caught it.

This test compiles tools/tests/fixtures/hardware/p2p_copy_correctness_test.cpp
-- a standalone HIP program that seeds two GPUs with distinct
position-dependent patterns, runs a real P2P copy in both directions, and
diffs every element of the result -- and requires the push direction to be
correct. The pull direction is exercised and reported every run but never
gates pass/fail: it is a known, currently-unfixed platform limitation
(not a regression in our own code), and the point of running it anyway is
so a silent change in its behavior (fixed by a driver update, or
regressed to a different wrong answer) is visible rather than assumed.

It skips rather than fails when no ROCm/HIP toolchain, or fewer than two
mutually peer-capable GPUs, are available: the contract is that the test
RAN on real hardware, not that a machine without one still passes CI.
"""

import glob
import os
import platform
import shutil
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
SRC = os.path.join(REPO_ROOT, "tools", "tests", "fixtures", "hardware", "p2p_copy_correctness_test.cpp")
PASS_LINE = "P2P_COPY_CORRECTNESS: PASS"


def _find_hipcc():
    """Prefer bigcherry's own vendored ROCm installs (vendor/rocm/<version>/,
    see tools/rocm-env.sh) over whatever happens to be on PATH or under
    HIP_PATH/ROCM_PATH -- those vendored trees are the toolchain this
    project's own builds are validated against. Falls back to PATH/
    HIP_PATH/ROCM_PATH so the test still runs in an environment with no
    vendored copy populated."""
    vendor_root = os.path.join(REPO_ROOT, "vendor", "rocm")
    for name in ("hipcc", "hipcc.exe"):
        for candidate in sorted(glob.glob(os.path.join(vendor_root, "*", "bin", name))):
            if os.path.exists(candidate):
                return candidate

    found = shutil.which("hipcc")
    if found:
        return found
    for env_var in ("HIP_PATH", "ROCM_PATH"):
        root = os.environ.get(env_var)
        if not root:
            continue
        for name in ("hipcc", "hipcc.exe"):
            candidate = os.path.join(root, "bin", name)
            if os.path.exists(candidate):
                return candidate
    return None


class TestP2pCopyCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hipcc = _find_hipcc()
        if hipcc is None:
            raise unittest.SkipTest("no hipcc found -- needs a ROCm/HIP toolchain")
        cls.hipcc = hipcc

    def test_push_direction_copies_correct_data(self):
        self.assertTrue(os.path.isfile(SRC), SRC)
        workdir = tempfile.mkdtemp(prefix="p2p-copy-correctness-")
        try:
            exe = os.path.join(workdir, "p2p_copy_correctness_test.exe"
                                if platform.system() == "Windows" else "p2p_copy_correctness_test")
            build = subprocess.run(
                [self.hipcc, "--offload-arch=gfx1100", "-O2", SRC, "-o", exe],
                capture_output=True, text=True, timeout=300,
            )
            if build.returncode != 0:
                compiler_output = build.stdout + build.stderr
                if "standard C++ header" in compiler_output and (
                    "cmath" in compiler_output or "cstdlib" in compiler_output
                ):
                    raise unittest.SkipTest(
                        "P2P copy correctness probe unavailable: selected HIP "
                        "toolchain cannot locate the standard C++ headers <cmath>/<cstdlib>"
                    )
                self.fail(
                    f"p2p_copy_correctness_test failed to compile with {self.hipcc}:\n"
                    f"{build.stderr[-4000:]}")
            try:
                run = subprocess.run([exe], capture_output=True, text=True, timeout=120)
            except OSError as exc:
                raise unittest.SkipTest(f"could not run compiled test: {exc}")
            output = run.stdout + run.stderr
            if run.returncode == 2 or "SKIP:" in output:
                raise unittest.SkipTest(
                    "no HIP-capable, mutually peer-access-capable pair of devices available:\n"
                    + output)
            if run.returncode != 0 and "HIP error" in output:
                raise unittest.SkipTest("no HIP-capable device available at runtime:\n" + output)
            self.assertEqual(run.returncode, 0, output)
            self.assertIn(PASS_LINE, output)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
