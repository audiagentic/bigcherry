"""Offline mechanics tests for 0300_mmq_forced_j on the composed b11552 source."""

from __future__ import annotations

import importlib.util
from collections import Counter
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import production_source  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_ID = "0300_mmq_forced_j"


def _load():
    spec = importlib.util.spec_from_file_location("patch_0300", _REPO / "engines/llamacpp/patches" / _ID / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()
_FILES = sorted({fp.path for fp in _P.PATCHES if not fp.create})
_CREATED = sorted({fp.path for fp in _P.PATCHES if fp.create})
_OWNED = {
    "ggml/src/ggml-cuda/bc-mmq-forced-j.cuh": (
        "ggml/src/ggml-cuda/mmq.cuh",
        '#include "bc-mmq-forced-j.cuh"  // BigCherry 0300 owned definitions\n\n',
        "static inline bool ggml_cuda_mmq_config_is_eligible(",
    ),
    "ggml/src/ggml-cuda/bc-mmq-forced-j-source.cuh": (
        "ggml/src/ggml-cuda/mmq.cu",
        '#include "bc-mmq-forced-j-source.cuh"  // BigCherry 0300 owned definitions\n\n',
        "bool ggml_cuda_mmq_variant_is_eligible(",
    ),
}


def _lines(*texts):
    return Counter(line.strip() for text in texts for line in text.splitlines() if line.strip())


@unittest.skipUnless(paths.llama_root().is_dir(), "pinned vendor checkout not present")
class Patch0300Mechanics(unittest.TestCase):
    def test_apply_on_composition_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = production_source(_FILES, td, stop_before=_ID)
            # no guard may already match: a guard that matches an earlier patch's output skips its edit silently
            before = {path: (root / path).read_text(encoding="utf-8") for path in _FILES}
            for fp in _P.PATCHES:
                if fp.create:
                    self.assertFalse((root / fp.path).exists())
                    continue
                for edit in fp.edits:
                    self.assertIsNone(re.search(edit.guard_pattern(), before[fp.path], re.M), (fp.path, edit.id))
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            texts = {path: (root / path).read_text(encoding="utf-8") for path in _FILES + _CREATED}
            for fp in _P.PATCHES:
                for edit in fp.edits:
                    self.assertIsNotNone(re.search(edit.guard_pattern(), texts[fp.path], re.M), (fp.path, edit.id))
            self.assertEqual(set(_OWNED), set(_CREATED))
            for path, (upstream, include, definition) in _OWNED.items():
                self.assertIn(definition, texts[path])
                self.assertIn(include.strip(), texts[upstream])
                self.assertNotIn(definition, texts[upstream])
            self.assertIn("ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst, 0);", texts["ggml/src/ggml-cuda/ggml-cuda.cu"])
            self.assertIn("int bc_forced_J = forced_J;", texts["ggml/src/ggml-cuda/mmq.cu"])
            self.assertIn("ggml_cuda_mul_mat_q_variant(", texts["ggml/src/ggml-cuda/bc-mmq-forced-j-source.cuh"])
            legacy = {path: texts[path] for path in _FILES}
            for path, (upstream, include, _) in _OWNED.items():
                legacy[upstream] = legacy[upstream].replace(include, texts[path], 1)
            new_lines = _lines(*(texts[p] for p in _FILES), *(texts[p] for p in _CREATED))
            old_lines = _lines(*(legacy[p] for p in _FILES))
            self.assertEqual(new_lines, old_lines + _lines(*(v[1] for v in _OWNED.values())))
            self.assertEqual(sum(new_lines.values()), sum(old_lines.values()) + 2)
            # HI71: J is forced before upstream sizes the activation padding from it
            mmq = texts["ggml/src/ggml-cuda/mmq.cu"]
            self.assertLess(mmq.index("J_best = bc_forced_J;"),
                            mmq.index("const size_t src1_q8_1_padding = ((J_best * sizeof(block_q8_1_mmq)"))
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            for path, text in texts.items():
                self.assertEqual(text, (root / path).read_text(encoding="utf-8"))

    def test_missing_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = production_source(_FILES, td, stop_before=_ID)
            # the first upstream file without the text its edits attach to
            target = root / "ggml/src/ggml-cuda/mmq.cuh"
            target.write_text("// moved upstream" + chr(10), encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
