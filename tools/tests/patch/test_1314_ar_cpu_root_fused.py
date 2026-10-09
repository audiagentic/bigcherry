"""Offline mechanics tests for 1314_ar_cpu_root_fused (pinned ggml-cuda.cu with 0840 + 0860 + 1291 applied)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_VENDOR = paths.llama_root()
_REL = "ggml/src/ggml-cuda/ggml-cuda.cu"


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "engines" / "llamacpp" / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = [_load(p) for p in ("0860_allreduce_provider_cli", "1225_hi85_nccl_heterogeneous_arch_guard", "0840_hybrid_allreduce_dispatch", "1291_ar_cpu_root")]
_P1314 = _load("1314_ar_cpu_root_fused")


def _only_cuda(patches):
    return [p for p in patches if p.path == _REL]


@unittest.skipUnless((_VENDOR / _REL).exists(), "pinned vendor checkout not present")
class Patch1314Mechanics(unittest.TestCase):
    def test_apply_after_1291_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / _REL
            path.parent.mkdir(parents=True)
            copy_pinned(_VENDOR / _REL, path)
            for mod in _BASE:
                res = apply_all(_only_cuda(mod.PATCHES), root)
                self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            results = apply_all(_P1314.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            kernel = out.index("static __global__ void bc_cpu_root_fused(")
            self.assertLess(out.index("static __global__ void bc_cpu_root_consume("), kernel)
            self.assertLess(kernel, out.index("static void bc_cpu_root_worker(bc_cpu_root * cr) {"))
            launch = out.index("bigcherry 1314: one launch per rank")
            # the unfused pair stays as the else branch
            self.assertLess(launch, out.index("bc_cpu_root_produce<<<1, 1024, 0, stream>>>", launch))
            self.assertEqual(out.count("bc_cpu_root_fused<<<"), 1)
            second = apply_all(_P1314.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
