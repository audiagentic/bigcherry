"""Offline mechanics tests for 1294_topk_deterministic_ties on the composed production source."""

from __future__ import annotations

from collections import Counter
import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patch.pinned_source import production_source  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PID = "1294_topk_deterministic_ties"
_REL = "ggml/src/ggml-cuda/top-k.cu"
_BC = "ggml/src/ggml-cuda/bc-topk-ties.cuh"
_INPUTS = [_REL]


def _load():
    spec = importlib.util.spec_from_file_location("patch_1294", _REPO / "engines/llamacpp/patches" / _PID / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


def _lines(*texts: str) -> Counter[str]:
    return Counter("".join(line.split()) for text in texts for line in text.splitlines() if line.strip())


@unittest.skipUnless(paths.llama_root().is_dir(), "pinned vendor checkout not present")
class Patch1294Mechanics(unittest.TestCase):
    def _root(self, td):
        return production_source(_INPUTS, Path(td), stop_before=_PID)

    def test_apply_composed_source_owned_file_guards_identity_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            before = (root / _REL).read_text(encoding="utf-8")

            for fp in _P.PATCHES:
                if fp.path == _BC:
                    continue
                for edit in fp.edits:
                    self.assertIsNone(re.search(edit.guard, before, re.M), (fp.path, edit.id))

            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            out = (root / _REL).read_text(encoding="utf-8")
            owned = (root / _BC).read_text(encoding="utf-8")
            self.assertEqual(owned, _P._TIE_KERNEL)
            self.assertEqual(sum(1 for line in owned.splitlines() if line.strip()), 142)

            include_text = '#include "bc-topk-ties.cuh"  // BigCherry 1294 implementation\n\n'
            self.assertEqual(out.count('#include "bc-topk-ties.cuh"'), 1)
            for definition in (
                "static __global__ void top_k_radix_gather_ties(",
                "static __global__ void top_k_radix_gather_ordered(",
                "static bool top_k_bc_ordered()",
                "static bool top_k_bc_deterministic_ties()",
            ):
                self.assertNotIn(definition, out)
                self.assertIn(definition, owned)

            include = out.index('#include "bc-topk-ties.cuh"')
            state = out.rfind("top_k_radix_state", 0, include)
            ordered_key = out.rfind("top_k_float_to_ordered", 0, include)
            self.assertGreaterEqual(state, 0)
            self.assertGreaterEqual(ordered_key, 0)
            launcher = out.index("static void top_k_radix_cuda(", include)
            self.assertLess(state, include)
            self.assertLess(ordered_key, include)
            self.assertLess(include, launcher)

            launch_ordered = out.index("top_k_radix_gather_ordered<BLOCK_SIZE><<<iter_nrows", launcher)
            launch_ties = out.index("top_k_radix_gather_ties<BLOCK_SIZE><<<iter_nrows", launch_ordered)
            self.assertLess(launch_ordered, launch_ties)
            self.assertLess(out.index("const bool det_ties = top_k_bc_deterministic_ties();", launcher), launch_ordered)

            legacy = out.replace(include_text, _P._TIE_KERNEL, 1)
            legacy_lines = _lines(legacy)
            new_lines = _lines(out, owned)
            include_line = _lines(include_text)
            self.assertEqual(new_lines, legacy_lines + include_line)
            self.assertEqual(sum(new_lines.values()), sum(legacy_lines.values()) + 1)

            for fp in _P.PATCHES:
                text = owned if fp.path == _BC else out
                for edit in fp.edits:
                    self.assertIsNotNone(re.search(edit.guard, text, re.M), (fp.path, edit.id))

            snapshot = {_REL: out, _BC: owned}
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(snapshot, {p: (root / p).read_text(encoding="utf-8") for p in snapshot})

    def test_missing_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            path = root / _REL
            before = path.read_text(encoding="utf-8")
            self.assertEqual(before.count("static void top_k_radix_cuda("), 1)
            drifted = before.replace("static void top_k_radix_cuda(", "static void top_k_radix_cuda_drifted(", 1)
            path.write_text(drifted, encoding="utf-8", newline="\n")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(drifted, path.read_text(encoding="utf-8"))
            self.assertFalse((root / _BC).exists())


if __name__ == "__main__":
    unittest.main()
