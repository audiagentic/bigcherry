"""Offline mechanics tests for 1294_topk_deterministic_ties (pinned ggml/src/ggml-cuda/top-k.cu)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_REL = "ggml/src/ggml-cuda/top-k.cu"
_VENDOR = _REPO / "vendor/llama.cpp" / _REL


def _load():
    spec = importlib.util.spec_from_file_location("patch_1294", _REPO / "patches/1294_topk_deterministic_ties/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1294Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / _REL).parent.mkdir(parents=True)
            shutil.copy2(_VENDOR, root / _REL)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            out = (root / _REL).read_text(encoding="utf-8")
            vendor = _VENDOR.read_text(encoding="utf-8")
            # the upstream gather (including its atomic equal branch) is untouched
            gather = vendor[vendor.index("static __global__ void top_k_radix_gather("):]
            gather = gather[:gather.index("\n}\n") + 3]
            self.assertIn(gather, out)
            # the tie kernel only rewrites ambiguous cutoffs, and runs after the gather in stream order
            self.assertIn("if (st.rank <= 0 || st.equal_count <= st.rank) {", out)
            launch = out.index("top_k_radix_gather_ties<BLOCK_SIZE><<<nrows")
            self.assertLess(out.index("src, dst, states, ncols, k, blocks_per_row);"), launch)
            self.assertLess(out.index("const bool det_ties = top_k_bc_deterministic_ties();"), launch)
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second))
            self.assertEqual(out, (root / _REL).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
