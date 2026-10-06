"""Offline mechanics tests for 1340_meta_per_device_arena."""

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
_PIN = "d89651a7b205"
_META = "ggml/src/ggml-backend-meta.cpp"
_BACKEND = "ggml/src/ggml-backend.cpp"
_MODEL = "src/llama-model.cpp"
_FATTN = "ggml/src/ggml-cuda/fattn.cu"


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


_P1283 = _load("patch_1283", _REPO / "patches/1283_qwen4exp_expert_parallel/patch.py")
_P1303 = _load("patch_1303", _REPO / "patches/1303_attn_kv_tensor_split/patch.py")
_P1326 = _load("patch_1326", _REPO / "patches/1326_sched_async_host_inputs/patch.py")
_P1336 = _load("patch_1336", _REPO / "patches/1336_sched_copy_callback/patch.py")
_P1339 = _load("patch_1339", _REPO / "patches/1339_meta_memory_report/patch.py")
_P = _load("patch_1340", _REPO / "patches/1340_meta_per_device_arena/patch.py")
_SRC = {path: _pinned(path) for path in (_META, _BACKEND, _MODEL, _FATTN)}


@unittest.skipUnless(all(text is not None for text in _SRC.values()), "pinned vendor repository not present")
class Patch1340Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        for path, text in _SRC.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text, encoding="utf-8", newline="\n")
        base = apply_all(_P1339.PATCHES, root)
        self.assertTrue(all(r.ok for r in base), [e.detail for r in base for e in r.failed])
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            meta = (root / _META).read_text(encoding="utf-8")
            backend = (root / _BACKEND).read_text(encoding="utf-8")

            self.assertIn('getenv("BIGCHERRY_META_PER_DEVICE_ARENA")', meta)
            self.assertIn("ggml_gallocr_ptr                     arena_galloc;", meta)
            self.assertIn("bc.arena_galloc.reset();", meta)
            self.assertLess(meta.index("bc.arena_galloc.reset();"), meta.index("ggml_backend_free(bc.backend);"))
            self.assertIn("bool ggml_backend_meta_alloc_graph(", meta)
            self.assertIn("ggml_gallocr_reserve(bcj.arena_galloc.get(), &simple_graph)", meta)
            self.assertIn("ggml_gallocr_alloc_graph(bcj.arena_galloc.get(), &simple_graph)", meta)
            # zero-sized slices are external whether deferred (no buffer) or static (dummy buffer, no data)
            self.assertIn("if (ret->data == nullptr && ret->view_src == nullptr && ggml_nelements(ret) == 0) {", meta)
            self.assertIn("GGML_ASSERT(ret->buffer != nullptr || (ret->flags & GGML_TENSOR_FLAG_COMPUTE) == 0);", meta)
            self.assertIn("ret->data = t->data; // Meta's fake logical address: allocator sentinel only", meta)
            self.assertIn("BIGCHERRY_META_MEM arena dev=%zu buft=%s size_mib=%.2f", meta)
            self.assertIn("BIGCHERRY_META_MEM arena_phase dev=%zu phase=reserve_begin", meta)
            self.assertIn("BIGCHERRY_META_MEM arena_phase dev=%zu phase=reserve_end", meta)
            self.assertIn("BIGCHERRY_META_MEM arena_phase dev=%zu phase=alloc_begin", meta)
            self.assertIn("BIGCHERRY_META_MEM arena_phase dev=%zu phase=alloc_end", meta)
            self.assertIn("bufs.resize(n_simple_bufts, nullptr);", meta)
            self.assertIn("if (t_ij->view_src->data != nullptr)", meta)
            self.assertIn("ggml_backend_meta_alloc_graph(sched->backends[i], &sched->graph)", backend)
            self.assertIn("reserve must instantiate the logical Meta tensors once", backend)
            # a reserve is not followed by a compute, so it rotates the simple-tensor containers itself
            self.assertIn("void ggml_backend_meta_rotate_graph_containers(struct ggml_cgraph * cgraph) {", meta)
            self.assertLess(backend.index("reserve must instantiate the logical Meta tensors once"), backend.index("ggml_backend_meta_rotate_graph_containers(&sched->graph);"))

            # the definitions, not the forward declarations near the top of the file
            arena = meta[meta.rindex("static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer("):
                         meta.rindex("static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(")]
            self.assertLess(arena.index("if (ggml_backend_meta_per_device_arena_enabled())"),
                            arena.index("bufs.push_back(ggml_backend_buft_alloc_buffer("))
            self.assertIn("BIGCHERRY_META_MEM compute dev=", arena)

            before = {p: (root / p).read_text(encoding="utf-8") for p in _SRC}
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before, {p: (root / p).read_text(encoding="utf-8") for p in _SRC})

    def test_production_meta_backend_edits_compose(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for src_path, text in _SRC.items():
                (root / src_path).parent.mkdir(parents=True, exist_ok=True)
                (root / src_path).write_text(text, encoding="utf-8", newline="\n")

            # These are the validated patches in the owner's current Meta/Qwen4Exp layout that edit the same files.
            # Restrict each package to these three source files; unrelated package edits have their own mechanics tests.
            for patch in (_P1283, _P1303, _P1326, _P1336, _P1339, _P):
                relevant = [fp for fp in patch.PATCHES if fp.path in _SRC]
                res = apply_all(relevant, root)
                self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            meta = (root / _META).read_text(encoding="utf-8")
            backend = (root / _BACKEND).read_text(encoding="utf-8")
            model = (root / _MODEL).read_text(encoding="utf-8")
            self.assertIn("BigCherry 1283: whole-expert MoE block.", meta)
            self.assertIn("BIGCHERRY_ATTN_TS", model)
            self.assertIn("bigcherry 1326", backend)
            self.assertIn("BigCherry 1336", backend)
            self.assertIn("BIGCHERRY_META_MEM compute dev=", meta)
            self.assertIn("ggml_gallocr_ptr                     arena_galloc;", meta)
            self.assertIn("failed to allocate per-device Meta arena", backend)

    def test_zero_slice_bypasses_flash_attn_backend_sizing(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            fattn = (root / _FATTN).read_text(encoding="utf-8")
            self.assertIn("const int gqa_ratio = Q->ne[2] / K->ne[2];", fattn)

            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            meta = (root / _META).read_text(encoding="utf-8")
            helper = meta[meta.index("bool ggml_backend_meta_alloc_graph("):
                          len(meta)]
            self.assertIn("ggml_nelements(ret) == 0", helper)
            self.assertIn("GGML_ASSERT(ret->buffer != nullptr || (ret->flags & GGML_TENSOR_FLAG_COMPUTE) == 0);", helper)
            self.assertIn("ret->data = t->data;", helper)
            self.assertLess(helper.index("ret->data = t->data;"), helper.index("ggml_gallocr_reserve("))

    def test_changed_compute_allocator_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            path = root / _META
            src = path.read_text(encoding="utf-8")
            src = src.replace("    size_t max_size = 0;\n    std::vector<ggml_backend_buffer_t> bufs;\n",
                              "    size_t max_size = 1;\n    std::vector<ggml_backend_buffer_t> bufs;\n", 1)
            path.write_text(src, encoding="utf-8", newline="\n")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
