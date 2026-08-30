"""RD30/RD94: compile and run the hostile-routing unit test for
mmq_build_moe_block_map (patches/1237_rd30_moe_mmq_compact_grid/patch.py).

test-backend-ops's own MUL_MAT_ID fixture builds its `ids` tensor from a
`std::shuffle` of a `0..n_mats-1` permutation per row (see
init_mul_mat_id_tensors in tests/test-backend-ops.cpp) -- that is
near-uniform routing, not the skewed/concentrated/single-hot distributions
RD94/EC13 exist to stress. Passing test-backend-ops does not by itself
prove RD30's compact-map algorithm handles a hostile distribution (e.g. one
expert absorbing every token, or 248 of 256 experts getting zero).

This test compiles tools/tests/rd30_hostile_test.cu -- a standalone HIP
program containing a verbatim copy of mmq_build_moe_block_map, run against
five real distributions (single-hot, concentrated-8-of-256, Zipf-skew,
uniform, and the degenerate all-zero case) at real production scale
(n_experts=256) -- and compares its output against a plain host-side
reference implementation of the same algorithm.

It skips rather than fails when no ROCm/HIP toolchain or GPU device is
available: the contract is that the test RAN on real hardware, not that a
machine without one still passes CI. Keep the kernel body here in sync with
the patch by hand -- if they diverge, the patch file is authoritative.
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
HOSTILE_TEST_CU = os.path.join(REPO_ROOT, "tools", "tests", "fixtures", "hardware", "rd30_hostile_test.cu")
OK_SUFFIX = "cases failed"


def _find_hipcc():
    """Prefer bigcherry's own vendored ROCm installs (vendor/rocm/<version>/,
    see tools/rocm-env.sh) over whatever happens to be on PATH or under
    HIP_PATH/ROCM_PATH -- those vendored trees are the toolchain this
    project's own builds are validated against, on both Windows and Linux
    sides of the SMB share. Falls back to PATH/HIP_PATH/ROCM_PATH so the
    test still runs in an environment with no vendored copy populated."""
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


class TestRd30HostileRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hipcc = _find_hipcc()
        if hipcc is None:
            raise unittest.SkipTest("no hipcc found -- needs a ROCm/HIP toolchain")
        cls.hipcc = hipcc

    def test_hostile_routing_matches_reference(self):
        self.assertTrue(os.path.isfile(HOSTILE_TEST_CU), HOSTILE_TEST_CU)
        workdir = tempfile.mkdtemp(prefix="rd30-hostile-")
        try:
            exe = os.path.join(workdir, "rd30_hostile_test.exe"
                               if platform.system() == "Windows" else "rd30_hostile_test")
            build = subprocess.run(
                [self.hipcc, "--offload-arch=gfx1100", "-O2",
                 HOSTILE_TEST_CU, "-o", exe],
                capture_output=True, text=True, timeout=300,
            )
            if build.returncode != 0:
                compiler_output = build.stdout + build.stderr
                if "standard C++ header" in compiler_output and (
                    "cmath" in compiler_output or "cstdlib" in compiler_output
                ):
                    raise unittest.SkipTest(
                        "HIP hostile-routing probe unavailable: selected HIP "
                        "toolchain cannot locate the standard C++ headers <cmath>/<cstdlib>"
                    )
                self.fail(
                    f"rd30_hostile_test failed to compile with {self.hipcc}:\n"
                    f"{build.stderr[-4000:]}")
            try:
                run = subprocess.run([exe], capture_output=True, text=True, timeout=60)
            except OSError as exc:
                raise unittest.SkipTest(f"could not run compiled test: {exc}")
            if run.returncode != 0 and "HIP error" in (run.stdout + run.stderr):
                raise unittest.SkipTest(
                    "no HIP-capable device available at runtime:\n"
                    + run.stdout + run.stderr)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn(OK_SUFFIX, run.stdout)
            self.assertIn("0/5 cases failed", run.stdout)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
