"""Exact candidate replay and the host B16 runtime contract fixture."""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcherry.patch import patchset
from bigcherry.patcher import apply_all
from _vtp_promotion_support import CANDIDATE_COMMIT, PATCHES, assert_promoted_recipe, load_patch_files
from test_vtp14_n1_rank_code_selection import radiance_checkout

PACKAGE = "rad0050_b16_kv_runtime_contract"
ABI44 = "rad0048_abi44_kv_contract"
FIXTURE = "bc_b16_kv_runtime_test"


def candidate_modules() -> list[str]:
    closure = patchset.expand_composition("radiance", (PACKAGE,), directory=PATCHES)
    resolved = patchset.resolve_exact("radiance", closure.expanded, directory=PATCHES)
    return [patch.patch_id for patch in resolved.modules]


class B16RuntimeContractTests(unittest.TestCase):
    def test_promoted_dependency(self):
        self.assertEqual(candidate_modules()[-2:], [ABI44, PACKAGE])
        assert_promoted_recipe(self, ABI44, "rad0049_qwen4exp_e4m3_kv", PACKAGE)

    def test_exact_candidate_replay_and_cpu_runtime_fixture(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("set RADIANCE_SRC to a checkout with the frozen ABI43 commit")
        if not all(shutil.which(tool) for tool in ("cmake", "ctest", "ninja")):
            self.skipTest("the host fixture requires CMake, CTest, and Ninja")

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temp:
            root = Path(temp) / "radiance"
            root.mkdir()
            archive = subprocess.check_output(
                ["git", "-C", str(checkout), "archive", "--format=tar", CANDIDATE_COMMIT],
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(root, filter="data")

            for module in candidate_modules()[:-1]:
                prefix = apply_all(load_patch_files(module), root)
                self.assertTrue(prefix and all(result.ok for result in prefix), (module, prefix))

            files = load_patch_files(PACKAGE)
            first = apply_all(files, root)
            self.assertTrue(first and all(result.ok for result in first), first)
            self.assertTrue(all(edit.status == "applied" for result in first for edit in result.results), first)
            snapshot = {patch.path: (root / patch.path).read_bytes() for patch in files}
            replay = apply_all(files, root)
            self.assertTrue(replay and all(result.ok for result in replay), replay)
            self.assertTrue(
                all(edit.status == "already-applied" for result in replay for edit in result.results), replay,
            )
            self.assertEqual(snapshot, {patch.path: (root / patch.path).read_bytes() for patch in files})

            build = root.parent / "host"
            subprocess.run(
                ["cmake", "-S", str(root), "-B", str(build), "-G", "Ninja",
                 "-DRAD_WITH_HIP=OFF", "-DRAD_WITH_FFMPEG=OFF",
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_FLAGS_RELEASE=-O0 -DNDEBUG",
                 "-DCMAKE_CXX_FLAGS_RELEASE=-O0 -DNDEBUG"],
                check=True,
            )
            subprocess.run(
                ["cmake", "--build", str(build), "--target", FIXTURE, "--parallel",
                 os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")],
                check=True,
            )
            subprocess.run(
                ["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error",
                 "-R", f"^{FIXTURE}$"],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
