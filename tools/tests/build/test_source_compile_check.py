from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.build import source_compile_check as scc  # noqa: E402


GOOD = r"""
static bool ggml_cuda_should_use_mmvf(
        int type, int cc, int warp_size, int ne, int nb, int ne11) {
    return type + cc + warp_size + ne + nb + ne11 > 0;
}

static __device__ constexpr int ggml_cuda_get_physical_warp_size() {
    return 64;
}

static __global__ void kernel_ok() {
    const int warp = ggml_cuda_get_physical_warp_size();
    (void) warp;
}

static void dispatch_ok() {
    (void) ggml_cuda_should_use_mmvf(1, 2, 32, 4, 5, 6);
}
"""

BAD_1347 = r"""
static bool ggml_cuda_should_use_mmvf(
        int type, int cc, int warp_size, int ne, int nb, int ne11) {
    return type + cc + warp_size + ne + nb + ne11 > 0;
}

static __device__ constexpr int ggml_cuda_get_physical_warp_size() {
    return 64;
}

static __global__ void kernel_ok() {
    const int warp = ggml_cuda_get_physical_warp_size();
    (void) warp;
}

// bigcherry 1347
static void dispatch_bad() {
    (void) ggml_cuda_should_use_mmvf(1, 2, 4, 5, 6);
}
"""

BAD_1345 = r"""
static bool ggml_cuda_should_use_mmvf(
        int type, int cc, int warp_size, int ne, int nb, int ne11) {
    return type + cc + warp_size + ne + nb + ne11 > 0;
}

static __device__ constexpr int ggml_cuda_get_physical_warp_size() {
    return 64;
}

static void dispatch_ok() {
    (void) ggml_cuda_should_use_mmvf(1, 2, 32, 4, 5, 6);
}

// bigcherry 1345
static void launch_bad() {
    const int warp = ggml_cuda_get_physical_warp_size();
    (void) warp;
}
"""


@unittest.skipUnless(shutil.which("clang++"), "clang++ required for source compile probes")
class SourceCompileCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="source-compile-check-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cuda = self.root / "ggml" / "src" / "ggml-cuda"
        self.cuda.mkdir(parents=True)
        self.work = self.root / "work"

    def _write(self, source: str) -> None:
        (self.cuda / "ggml-cuda.cu").write_text(source, encoding="utf-8")

    def test_good_composed_source_contracts_pass(self):
        self._write(GOOD)
        self.assertEqual(
            scc.check_cuda_contracts(self.root, self.work, "clang++"),
            [],
        )

    def test_historical_1347_missing_warp_size_is_rejected(self):
        self._write(BAD_1347)
        failures = scc.check_cuda_contracts(self.root, self.work, "clang++")
        rendered = "\n".join(failures)
        self.assertIn("ggml_cuda_should_use_mmvf signature mismatch", rendered)
        self.assertIn("owner=patch-1347", rendered)
        self.assertIn("args=5/6", rendered)

    def test_historical_1345_device_only_helper_host_call_is_rejected(self):
        self._write(BAD_1345)
        failures = scc.check_cuda_contracts(self.root, self.work, "clang++")
        rendered = "\n".join(failures)
        self.assertIn(
            "ggml_cuda_get_physical_warp_size is device-only but is called from host code",
            rendered,
        )
        self.assertIn("owner=patch-1345", rendered)

    def test_arity_parser_ignores_nested_commas(self):
        self.assertEqual(
            scc._arity("fn<int, 2>(x), std::array<int, 4>{}, false"),
            3,
        )


if __name__ == "__main__":
    unittest.main()
