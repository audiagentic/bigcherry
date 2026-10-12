"""Offline mechanics tests for 1355_hc_post_gate_fuse."""

from __future__ import annotations

import importlib.util
import re
from collections import Counter
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import production_source  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = paths.llama_root()  # the primary checkout's vendor tree (a slice worktree has none)
_FILES = (
    "ggml/src/ggml-cuda/dsv4-hc.cu",
    "ggml/src/ggml-cuda/dsv4-hc.cuh",
    "ggml/src/ggml-cuda/ggml-cuda.cu",
)
_OWNED = "ggml/src/ggml-cuda/bc-hc-post-gate-fuse.cuh"
_INCLUDE = '#include "bc-hc-post-gate-fuse.cuh"  // BigCherry 1355 owned definitions\n\n'


def _lines(*texts):
    return Counter(line.strip() for text in texts for line in text.splitlines() if line.strip())


def _load(pid: str):
    spec = importlib.util.spec_from_file_location(
        "patch_" + pid, _REPO / "engines" / "llamacpp" / "patches" / pid / "patch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1355_hc_post_gate_fuse")


@unittest.skipUnless(all((_V / p).exists() for p in _FILES), "pinned vendor checkout not present")
class Patch1355Mechanics(unittest.TestCase):
    def _root_with_production(self, td: str) -> Path:
        # the production set without 1355 (in the selection once validated; each test applies it itself)
        return production_source(_FILES, td, without="1355_hc_post_gate_fuse")

    def test_full_production_then_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root_with_production(td)
            cuda = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
            self.assertIn("bigcherry 1313: SCALE -> UNARY", cuda.read_text(encoding="utf-8"))
            self.assertIn("bigcherry 1344", (root / "ggml/src/ggml-cuda/dsv4-hc.cu").read_text(encoding="utf-8"))

            before = {p: (root / p).read_text(encoding="utf-8") for p in _FILES}
            self.assertFalse((root / _OWNED).exists())
            for fp in _P.PATCHES:
                if not fp.create:
                    for edit in fp.edits:
                        self.assertIsNone(re.search(edit.guard_pattern(), before[fp.path], re.MULTILINE), (fp.path, edit.id))

            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            src = (root / "ggml/src/ggml-cuda/dsv4-hc.cu").read_text(encoding="utf-8")
            owned = (root / _OWNED).read_text(encoding="utf-8")
            dispatch = cuda.read_text(encoding="utf-8")
            self.assertIn("bc_dsv4_hc_post_gate_grid_f32", owned)
            self.assertIn("const float gate_scaled = gate_s0 * gate_src", owned)
            self.assertIn(_INCLUDE.strip(), src)
            self.assertNotIn("static __global__ void bc_dsv4_hc_post_gate_grid_f32(", src)
            legacy = dict({p: (root / p).read_text(encoding="utf-8") for p in _FILES})
            legacy["ggml/src/ggml-cuda/dsv4-hc.cu"] = src.replace(_INCLUDE, owned, 1)
            new_lines = _lines(*[(root / p).read_text(encoding="utf-8") for p in _FILES])
            new_lines += _lines(owned)
            old_lines = _lines(*(legacy[p] for p in _FILES))
            self.assertEqual(new_lines, old_lines + _lines(_INCLUDE))
            self.assertEqual(sum(new_lines.values()), sum(old_lines.values()) + 1)
            self.assertIn("BIGCHERRY_HC_POST_GATE_FUSE", dispatch)
            self.assertIn("patch=1355_hc_post_gate_fuse", dispatch)
            self.assertLess(
                dispatch.index("BigCherry 1355 (QFP35)"),
                dispatch.index("bigcherry 1313: SCALE -> UNARY"),
            )

            before_second = {p: (root / p).read_text(encoding="utf-8") for p in (*_FILES, _OWNED)}
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before_second, {p: (root / p).read_text(encoding="utf-8") for p in before_second})

    def test_changed_1313_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root_with_production(td)
            p = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
            before = p.read_text(encoding="utf-8").replace(
                "// bigcherry 1313: SCALE -> UNARY(SILU|SIGMOID) [-> SCALE] in one launch",
                "// changed 1313 matcher",
                1,
            )
            p.write_text(before, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, p.read_text(encoding="utf-8"))

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_HC_POST_GATE_FUSE"])


if __name__ == "__main__":
    unittest.main()
