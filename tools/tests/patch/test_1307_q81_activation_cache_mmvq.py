"""Offline mechanics tests for 1307_q81_activation_cache_mmvq (pinned mmvq.cu after 0600 + 1241; ggml-cuda.cu)."""

from __future__ import annotations

import importlib.util
import re
from collections import Counter
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned, production_source  # noqa: E402
from bigcherry.core import paths  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_LLAMA = paths.llama_root()
_VENDOR = _LLAMA / "ggml/src/ggml-cuda"  # pinned source (copy_pinned reads the HEAD commit)
_GGML_CUDA = _LLAMA / "ggml/src/ggml-cuda/ggml-cuda.cu"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load("patch_1307", _REPO / "engines/llamacpp/patches/1307_q81_activation_cache_mmvq/patch.py")
_PATCHES = [p for p in _module.PATCHES if p.path in {"ggml/src/ggml-cuda/mmvq.cu", "ggml/src/ggml-cuda/ggml-cuda.cu"}]
_PREREQS = [
    _load("patch_0600", _REPO / "engines/llamacpp/patches/0600_mmvq_geometry/patch.py").PATCH,
    *_load("patch_1241", _REPO / "engines/llamacpp/patches/1241_rd33_mmvq_q8_0_f32_decode/patch.py").PATCHES,
]


@unittest.skipUnless(_GGML_CUDA.exists(), "pinned vendor checkout not present")
class Patch1307Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        for name in ("mmvq.cu", "mmvq.cuh", "vecdotq.cuh"):
            copy_pinned(_VENDOR / name, cuda / name)
        copy_pinned(_GGML_CUDA, cuda / "ggml-cuda.cu")
        prereq = apply_all(_PREREQS, root)
        assert all(r.ok for r in prereq), [e.detail for r in prereq for e in r.failed]
        return td, root, cuda

    def test_apply_and_idempotent(self):
        td, root, cuda = self._tree()
        with td:
            results = apply_all(_PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            mmvq = (cuda / "mmvq.cu").read_text(encoding="utf-8")
            cu = (cuda / "ggml-cuda.cu").read_text(encoding="utf-8")
            self.assertIn('#include "hip-q81-cache.h"', mmvq)
            self.assertIn("ggml_hip_q81_cache_find(q81, key)", mmvq)
            self.assertIn("ggml_hip_q81_cache_publish(q81, key, r);", mmvq)
            # flattened-reshape lookup (1312 GDN final_output) after the padding-free reshape lookup
            self.assertLess(mmvq.index("hit = ggml_hip_q81_cache_find(q81, vkey);"), mmvq.index("hit = ggml_hip_q81_cache_find(q81, fkey);"))
            self.assertIn("src0->data, src0->type, src1_q8_1_ptr, ids_d", mmvq)
            self.assertNotIn("src1_q8_1(ctx.pool(), ne13*ne12", mmvq)  # pool buffer only on fallback now
            # publish must follow the quantize that fills the reservation (1235's contract)
            q = mmvq.index("quantize_row_q8_1_cuda(src1_d, nullptr, r.ptr")
            self.assertLess(q, mmvq.index("ggml_hip_q81_cache_publish(q81, key, r);"))
            self.assertIn("ggml_hip_q81_cache_begin_generation(", cu)
            self.assertIn("ggml_hip_q81_cache_set_capture_active(ggml_hip_q81_cache_for_context(*cuda_ctx), true);", cu)
            self.assertIn("ggml_hip_q81_cache_set_capture_active(ggml_hip_q81_cache_for_context(*cuda_ctx), false);", cu)

            second = apply_all(_PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(mmvq, (cuda / "mmvq.cu").read_text(encoding="utf-8"))
            self.assertEqual(cu, (cuda / "ggml-cuda.cu").read_text(encoding="utf-8"))


# PA44-E packaging equivalence is part of the 1307 mechanics gate. Importing
# the TestCase here makes unittest's changed-patch loader execute the materialized
# pre/post merge tree checks instead of leaving the standalone regression undiscovered.
from tools.tests.patch.test_pa44e_merged_family_identity import PA44EMergedFamilyIdentity  # noqa: E402,F401


_ID = "1307_q81_activation_cache_mmvq"
_COMPOSE_INPUTS = sorted({fp.path for fp in _module.PATCHES if not fp.create})
_CREATED = sorted({fp.path for fp in _module.PATCHES if fp.create})
_OWNED = {
    "ggml/src/ggml-cuda/bc-rms-norm-q81.cuh": (
        "ggml/src/ggml-cuda/norm.cu",
        '#include "bc-rms-norm-q81.cuh"  // BigCherry 1309 owned definition\n\n',
        "static __global__ void rms_norm_mul_q81_f32(",
    ),
    "ggml/src/ggml-cuda/bc-act-q81.cuh": (
        "ggml/src/ggml-cuda/unary.cu",
        '#include "bc-act-q81.cuh"  // BigCherry 1310 owned definitions\n\n',
        "static bool bc_act_q81_try(",
    ),
    "ggml/src/ggml-cuda/bc-hc-pre-q81.cuh": (
        "ggml/src/ggml-cuda/dsv4-hc.cu",
        '#include "bc-hc-pre-q81.cuh"  // BigCherry 1311 owned definition\n\n',
        "static __global__ void dsv4_hc_pre_q81_f32(",
    ),
}


def _lines(*texts):
    return Counter(line.strip() for text in texts for line in text.splitlines() if line.strip())


@unittest.skipUnless(_GGML_CUDA.exists(), "pinned vendor checkout not present")
class Patch1307OwnedFiles(unittest.TestCase):
    def _root(self, td):
        return production_source(_COMPOSE_INPUTS, td, stop_before=_ID)

    def test_composed_source_owned_files_guards_identity_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            before = {p: (root / p).read_text(encoding="utf-8") for p in _COMPOSE_INPUTS}
            for fp in _module.PATCHES:
                if fp.create:
                    self.assertIn(fp.path, _CREATED)
                    self.assertFalse((root / fp.path).exists())
                    continue
                for edit in fp.edits:
                    self.assertEqual(edit.expect_matches, 1, (fp.path, edit.id))
                    self.assertIsNone(re.search(edit.guard_pattern(), before[fp.path], re.MULTILINE), (fp.path, edit.id))

            res = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            after = {p: (root / p).read_text(encoding="utf-8") for p in _COMPOSE_INPUTS}
            owned = {p: (root / p).read_text(encoding="utf-8") for p in _CREATED}

            self.assertEqual(set(_OWNED), set(_CREATED))
            for path, (upstream, include, definition) in _OWNED.items():
                self.assertIn(definition, owned[path])
                self.assertEqual(after[upstream].count(include.strip()), 1)
                self.assertNotIn(definition, after[upstream])

            self.assertIn("ggml_hip_q81_cache_publish(q81, key, r);", after["ggml/src/ggml-cuda/norm.cu"])
            self.assertIn("bc_act_q81_try<op, false>", after["ggml/src/ggml-cuda/unary.cu"])
            self.assertIn("dsv4_hc_pre_q81_f32<true>", after["ggml/src/ggml-cuda/dsv4-hc.cu"])
            self.assertIn("ggml_hip_q81_cache_begin_generation(", after["ggml/src/ggml-cuda/ggml-cuda.cu"])

            legacy = dict(after)
            for path, (upstream, include, _) in _OWNED.items():
                legacy[upstream] = legacy[upstream].replace(include, owned[path], 1)
            new_lines = _lines(*(after[p] for p in _COMPOSE_INPUTS), *(owned[p] for p in _CREATED))
            legacy_lines = _lines(*(legacy[p] for p in _COMPOSE_INPUTS))
            include_lines = _lines(*(v[1] for v in _OWNED.values()))
            self.assertEqual(new_lines, legacy_lines + include_lines)
            self.assertEqual(sum(new_lines.values()), sum(legacy_lines.values()) + 3)
            self.assertEqual(sum(sum(_lines(owned[p]).values()) for p in _CREATED), 180)

            texts = dict(after)
            texts.update(owned)
            for fp in _module.PATCHES:
                for edit in fp.edits:
                    self.assertIsNotNone(re.search(edit.guard_pattern(), texts[fp.path], re.MULTILINE), (fp.path, edit.id))

            snapshot = {p: (root / p).read_text(encoding="utf-8") for p in _COMPOSE_INPUTS + _CREATED}
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(snapshot, {p: (root / p).read_text(encoding="utf-8") for p in snapshot})

    def test_missing_owned_block_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            path = root / "ggml/src/ggml-cuda/norm.cu"
            source = path.read_text(encoding="utf-8")
            anchor = "static __global__ void rms_norm_back_f32("
            self.assertEqual(source.count(anchor), 1)
            path.write_text(source.replace(anchor, "static __global__ void rms_norm_back_f32_drifted(", 1), encoding="utf-8")
            snapshot = {p: (root / p).read_text(encoding="utf-8") for p in _COMPOSE_INPUTS}
            res = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(snapshot, {p: (root / p).read_text(encoding="utf-8") for p in _COMPOSE_INPUTS})
            for path in _CREATED:
                self.assertFalse((root / path).exists())


if __name__ == "__main__":
    unittest.main()
