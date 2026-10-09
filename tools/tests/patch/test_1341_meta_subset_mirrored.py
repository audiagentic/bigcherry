"""Offline mechanics tests for 1341_meta_subset_mirrored."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = _REPO / "vendor/llama.cpp"
_PIN = "HEAD"  # the vendor checkout is at the pinned revision
_H = "ggml/include/ggml-backend.h"
_META = "ggml/src/ggml-backend-meta.cpp"
_MODEL = "src/llama-model.cpp"
_BACKEND = "ggml/src/ggml-backend.cpp"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pinned(path):
    try:
        res = subprocess.run(["git", "-C", str(_V), "show", f"{_PIN}:{path}"], capture_output=True, text=True,
                             encoding="utf-8", check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout


_P1283 = _load("patch_1283", _REPO / "engines/llamacpp/patches/1283_qwen4exp_expert_parallel/patch.py")
_P1303 = _load("patch_1303", _REPO / "engines/llamacpp/patches/1303_attn_kv_tensor_split/patch.py")
_P1326 = _load("patch_1326", _REPO / "engines/llamacpp/patches/1326_sched_async_host_inputs/patch.py")
_P1339 = _load("patch_1339", _REPO / "engines/llamacpp/patches/1339_meta_memory_report/patch.py")
_P1340 = _load("patch_1340", _REPO / "engines/llamacpp/patches/1340_meta_per_device_arena/patch.py")
_P = _load("patch_1341", _REPO / "engines/llamacpp/patches/1341_meta_subset_mirrored/patch.py")
_SRC = {path: _pinned(path) for path in (_H, _META, _MODEL, _BACKEND, "ggml/src/ggml-cuda/ggml-cuda.cu")}


@unittest.skipUnless(all(text is not None for text in _SRC.values()), "pinned vendor repository not present")
class Patch1341Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        for path, text in _SRC.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text, encoding="utf-8", newline="\n")
        base = apply_all(_P1303.PATCHES, root)
        self.assertTrue(all(r.ok for r in base), [e.detail for r in base for e in r.failed])
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            hdr = (root / _H).read_text(encoding="utf-8")
            meta = (root / _META).read_text(encoding="utf-8")
            model = (root / _MODEL).read_text(encoding="utf-8")

            self.assertIn("uint32_t active_mask;", hdr)
            self.assertIn("auto merge_active_masks =", meta)
            self.assertIn("explicit subset masks must agree", meta)
            self.assertIn("split_state.active_mask = active_mask;", meta)
            self.assertIn("reset ne[0] on every device; the shape array is reused by this loop", meta)
            self.assertIn("ne[0] = ggml_backend_meta_split_device_active(split_state, j) ? tensor->ne[0] : 0;", meta)
            self.assertIn("if (split_state_src.active_mask != 0)", meta)
            self.assertIn("BIGCHERRY_META_SUBSET_MIRROR", model)
            self.assertIn("std::regex_match(tensor_name, pattern_idx_cache)", model)
            self.assertIn("bigcherry_attn_split.split[j] != 0.0f", model)
            self.assertIn("const size_t split_rotation =", model)
            # MSM03 step 2: the attention mask graph input follows the attention devices (behind its own flag)
            self.assertIn("static uint32_t bc_meta_attn_input_mask = 0;", meta)
            self.assertIn('strstr(tensor->name, "kq_mask") != nullptr', meta)
            self.assertLess(meta.index("static uint32_t bc_meta_attn_input_mask = 0;"), meta.index("split_state.active_mask = bc_meta_attn_input_mask;"))
            self.assertIn("BIGCHERRY_META_SUBSET_MIRROR_INPUTS", model)
            self.assertLess(model.index('extern "C" void ggml_backend_meta_set_attn_input_mask(uint32_t active_mask);'), model.index("ggml_backend_meta_set_attn_input_mask(split_state.active_mask);"))
            self.assertNotIn("BIGCHERRY_META_SUBSET_MIRROR", meta)

            # This phase seeds no compute-side name/pattern; only the persistent indexer-cache classifier.
            seed = model[model.index("BigCherry 1341 (MSM03): first subset-mirror seed"):
                         model.index("    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {")]
            self.assertIn("pattern_idx_cache", seed)
            self.assertNotIn("attn_inp_kq_mask", seed)
            self.assertNotIn("kpool", seed)

            before = {p: (root / p).read_text(encoding="utf-8") for p in _SRC}
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before, {p: (root / p).read_text(encoding="utf-8") for p in _SRC})

    def test_meta_memory_experiment_composes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for src_path, text in _SRC.items():
                (root / src_path).parent.mkdir(parents=True, exist_ok=True)
                (root / src_path).write_text(text, encoding="utf-8", newline="\n")

            for patch in (_P1283, _P1303, _P1326, _P1339, _P1340, _P):
                relevant = [fp for fp in patch.PATCHES if fp.path in _SRC]
                res = apply_all(relevant, root)
                self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            meta = (root / _META).read_text(encoding="utf-8")
            model = (root / _MODEL).read_text(encoding="utf-8")
            backend = (root / _BACKEND).read_text(encoding="utf-8")
            self.assertIn("std::vector<arena_plan_t>            arena_plans; // BigCherry 1340 (MSM02)", meta)
            self.assertIn("uint32_t active_mask", (root / _H).read_text(encoding="utf-8"))
            self.assertIn("BIGCHERRY_META_MEM arena dev=%zu", meta)
            self.assertIn("BIGCHERRY_META_SUBSET_MIRROR", model)
            self.assertIn("failed to allocate per-device Meta arena", backend)
            self.assertIn("BigCherry 1283: whole-expert MoE block.", meta)
            self.assertIn("bigcherry 1326", meta)

    def test_changed_seed_site_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            path = root / _MODEL
            src = path.read_text(encoding="utf-8")
            src = src.replace("    split_state.axis = tc.axis;\n", "    split_state.axis = (ggml_backend_meta_split_axis) tc.axis;\n", 1)
            path.write_text(src, encoding="utf-8", newline="\n")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
