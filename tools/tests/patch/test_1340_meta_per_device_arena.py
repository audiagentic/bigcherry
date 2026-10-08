"""Offline mechanics tests for 1340_meta_per_device_arena."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import tomllib
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = _REPO / "vendor/llama.cpp"
_PIN = "HEAD"  # the vendor checkout is at the pinned revision
_META = "ggml/src/ggml-backend-meta.cpp"
_BACKEND = "ggml/src/ggml-backend.cpp"
_ALLOC = "ggml/src/ggml-alloc.c"
_ALLOC_H = "ggml/include/ggml-alloc.h"
_META_H = "ggml/include/ggml-backend.h"
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
_MERGED = _load("patch_1340", _REPO / "patches/1340_meta_per_device_arena/patch.py")
_P1339 = SimpleNamespace(PATCHES=[p for p in _MERGED.PATCHES if p.description.startswith("1339:")])
_P = SimpleNamespace(PATCHES=[p for p in _MERGED.PATCHES if p.description.startswith("1340:")])
_P1341 = _load("patch_1341", _REPO / "patches/1341_meta_subset_mirrored/patch.py")
_SRC = {path: _pinned(path) for path in (
    _META, _BACKEND, _ALLOC, _ALLOC_H, _META_H, _MODEL, _FATTN, "ggml/src/ggml-cuda/ggml-cuda.cu"
)}


@unittest.skipUnless(all(text is not None for text in _SRC.values()), "pinned vendor repository not present")
class Patch1340Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        for path, text in _SRC.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text, encoding="utf-8", newline="\n")
        # 1340 is validated against the experiment stack in its real application order.
        for patch in (_P1283, _P1303, _P1326, _P1339, _P1341):
            relevant = [fp for fp in patch.PATCHES if fp.path in _SRC]
            base = apply_all(relevant, root)
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
            self.assertIn("std::vector<arena_plan_t>            arena_plans; // BigCherry 1340 (MSM02)", meta)
            self.assertIn("std::vector<ggml_tensor *>           arena_nodes;", meta)
            self.assertIn("std::vector<ggml_tensor *>           arena_leafs;", meta)
            self.assertIn("ggml_backend_meta_buffer_simple_tensors", meta)
            self.assertNotIn("std::vector<ggml_tensor *> nodes(cgraph->n_nodes);", meta)
            self.assertNotIn("std::vector<ggml_tensor *> leafs(cgraph->n_leafs);", meta)
            self.assertIn("bc.arena_plans.clear();", meta)
            self.assertLess(meta.index("bc.arena_plans.clear();"), meta.index("ggml_backend_free(bc.backend);"))
            self.assertIn("bool ggml_backend_meta_alloc_graph(", meta)
            self.assertNotIn('getenv("BIGCHERRY_META_ARENA_FASTBIND")', meta)
            # reserve-time shape plans own metadata only; one per-device gallocr owns physical storage
            self.assertIn("struct arena_plan_t {", meta)
            self.assertIn("ggml_gallocr_ptr                    arena_galloc;", meta)
            self.assertIn("ggml_gallocr_reserve_n_size(bcj.arena_plans[i_plan].galloc.get()", meta)
            self.assertIn("ggml_gallocr_reserve_grow(bcj.arena_galloc.get(), &simple_graph)", meta)
            self.assertIn("ggml_gallocr_alloc_graph_reuse_from(plan, bcj.arena_galloc.get(), &simple_graph)", meta)
            self.assertNotIn("logical_replanned", meta)
            # zero-sized slices are external whether deferred (no buffer) or static (dummy buffer, no data)
            self.assertIn("if (is_meta && ret->data == nullptr && ret->view_src == nullptr && ggml_nelements(ret) == 0) {", meta)
            self.assertNotIn("GGML_TENSOR_FLAG_COMPUTE) == 0);", meta)
            self.assertIn("ret->data = t->data; // Meta's fake logical address: allocator sentinel only", meta)
            self.assertIn('"tensor=%s op=%s need=%zu planned=%zu"', (root / _ALLOC).read_text(encoding="utf-8"))
            self.assertIn("BIGCHERRY_META_MEM arena dev=%zu buft=%s reserved_mib=%.2f plans=%zu replans=%llu", meta)
            self.assertIn("uint64_t                             arena_replans = 0;", meta)
            self.assertIn("bcj.arena_replans++;", meta)
            self.assertIn("BIGCHERRY_META_MEM arena_replan dev=%zu", meta)
            self.assertIn("BIGCHERRY_META_MEM arena_grew dev=%zu", meta)
            self.assertIn("ggml_gallocr_reserve_n_size(", meta)
            self.assertIn("ggml_gallocr_reserve_grow(bcj.arena_galloc.get(), &simple_graph)", meta)
            self.assertNotIn("arena_phase dev=", meta)
            self.assertIn("bufs.resize(n_simple_bufts, nullptr);", meta)
            self.assertIn("if (t_ij->view_src->data != nullptr)", meta)
            self.assertIn("ggml_backend_meta_alloc_graph(sched->backends[i], &sched->graph)", backend)
            self.assertIn("ggml_backend_meta_reserve_graph(sched->backends[i], &sched->graph)", backend)
            self.assertNotIn("logical_replanned", backend)
            self.assertIn("bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph)",
                          (root / _ALLOC).read_text(encoding="utf-8"))
            alloc_src = (root / _ALLOC).read_text(encoding="utf-8")
            # the definition must lose its `static` (a guard that matched the original text once skipped this edit
            # and the build failed: static declaration follows non-static declaration)
            self.assertNotIn("static bool ggml_gallocr_needs_realloc(", alloc_src)
            self.assertEqual(alloc_src.count("bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {"), 1)
            alloc_hdr = (root / _ALLOC_H).read_text(encoding="utf-8")
            self.assertIn("GGML_API bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph);", alloc_hdr)
            self.assertIn("GGML_API bool ggml_gallocr_reserve_grow(ggml_gallocr_t galloc, struct ggml_cgraph * graph);", alloc_hdr)
            self.assertIn("GGML_API bool ggml_gallocr_alloc_graph_reuse_from(", alloc_hdr)
            self.assertIn("bool ggml_gallocr_alloc_graph_reuse(ggml_gallocr_t galloc, struct ggml_cgraph * graph)", alloc_src)
            self.assertIn("bool ggml_gallocr_reserve_grow(ggml_gallocr_t galloc, struct ggml_cgraph * graph)", alloc_src)
            self.assertIn("bool ggml_gallocr_alloc_graph_reuse_from(", alloc_src)
            self.assertIn("return ggml_gallocr_alloc_graph_reuse(galloc, graph);", alloc_src)
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

            # These are the validated patches in the owner's current Meta/Qwen4Exp layout that edit the fixture files.
            # Unrelated package edits have their own mechanics tests.
            for patch in (_P1283, _P1303, _P1326, _P1339, _P1341, _P):
                relevant = [fp for fp in patch.PATCHES if fp.path in _SRC]
                res = apply_all(relevant, root)
                self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            meta = (root / _META).read_text(encoding="utf-8")
            backend = (root / _BACKEND).read_text(encoding="utf-8")
            model = (root / _MODEL).read_text(encoding="utf-8")
            self.assertIn("BigCherry 1283: whole-expert MoE block.", meta)
            self.assertIn("BIGCHERRY_ATTN_TS", model)
            self.assertIn("bigcherry 1326", backend)
            self.assertIn("BIGCHERRY_META_MEM compute dev=", meta)
            self.assertIn("ggml_backend_meta_split_device_active", meta)
            self.assertIn("std::vector<arena_plan_t>            arena_plans; // BigCherry 1340 (MSM02)", meta)
            self.assertIn("std::vector<ggml_tensor *>           arena_nodes;", meta)
            self.assertIn("failed to allocate per-device Meta arena", backend)
            self.assertIn("reserved_mib=%.2f plans=%zu replans=%llu", meta)
            self.assertNotIn("logical_replanned", backend)

    def test_zero_slice_bypasses_flash_attn_backend_sizing(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            fattn = (root / _FATTN).read_text(encoding="utf-8")
            self.assertIn("const int gqa_ratio = Q->ne[2] / K->ne[2];", fattn)

            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            meta = (root / _META).read_text(encoding="utf-8")
            helper = meta[meta.index("static void ggml_backend_meta_arena_map_graph("):
                          len(meta)]
            self.assertIn("ggml_nelements(ret) == 0", helper)
            self.assertNotIn("GGML_TENSOR_FLAG_COMPUTE) == 0);", helper)
            self.assertIn("ret->data = t->data;", helper)
            self.assertLess(helper.index("ret->data = t->data;"), helper.index("ggml_gallocr_reserve_n_size("))

    def test_edit_contracts_and_guards_are_post_edit_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            before = {path: (root / path).read_text(encoding="utf-8") for path in _SRC}

            for file_patch in _P.PATCHES:
                self.assertIn(file_patch.path, before)
                for edit in file_patch.edits:
                    self.assertEqual(edit.expect_matches, 1, edit.id)
                    self.assertIsNotNone(edit.guard, edit.id)
                    self.assertGreater(edit.max_span_lines, 0, edit.id)
                    self.assertIsNone(
                        re.search(edit.guard_pattern(), before[file_patch.path], re.MULTILINE),
                        f"{edit.id}: guard already matches the pre-1340 source",
                    )

            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            after = {path: (root / path).read_text(encoding="utf-8") for path in _SRC}
            for file_patch in _P.PATCHES:
                for edit in file_patch.edits:
                    self.assertIsNotNone(
                        re.search(edit.guard_pattern(), after[file_patch.path], re.MULTILINE),
                        f"{edit.id}: guard does not identify its post-edit output",
                    )

    def test_reserve_only_growth_and_nonfit_fallback_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            meta = (root / _META).read_text(encoding="utf-8")
            backend = (root / _BACKEND).read_text(encoding="utf-8")
            alloc = (root / _ALLOC).read_text(encoding="utf-8")

            reserve = meta[
                meta.index("bool ggml_backend_meta_reserve_graph("):
                meta.index("// Compute-time only:", meta.index("bool ggml_backend_meta_reserve_graph("))
            ]
            compute = meta[
                meta.index("bool ggml_backend_meta_alloc_graph("):
                meta.index("// BigCherry 1340 (MSM02): what graph_compute does", meta.index("bool ggml_backend_meta_alloc_graph("))
            ]

            self.assertIn("ggml_gallocr_reserve_n_size(", reserve)
            self.assertIn("ggml_gallocr_reserve_grow(", reserve)
            self.assertIn("ggml_gallocr_needs_realloc(", compute)
            self.assertIn("if (nonfit) {", compute)
            # a new layout inside the reserved arena is quiet; only growth after reserve is an error
            self.assertIn("if (after_bytes > before_bytes) {", compute)
            self.assertLess(compute.index("if (after_bytes > before_bytes) {"), compute.index("GGML_LOG_ERROR("))
            self.assertIn("the reserve-time arena was not the worst case", compute)
            self.assertEqual(compute.count("ggml_gallocr_reserve_grow("), 1)
            self.assertLess(compute.index("if (nonfit) {"), compute.index("ggml_gallocr_reserve_grow("))
            self.assertLess(compute.index("ggml_gallocr_reserve_grow("), compute.index("ggml_gallocr_alloc_graph_reuse_from("))
            self.assertNotIn("ggml_gallocr_alloc_graph(plan", compute)
            self.assertIn("bool ggml_gallocr_reserve_grow(", alloc)
            self.assertIn("bool ggml_gallocr_alloc_graph_reuse_from(", alloc)
            self.assertIn("ggml_backend_meta_reserve_graph(sched->backends[i], &sched->graph)", backend)
            self.assertIn("ggml_backend_meta_alloc_graph(sched->backends[i], &sched->graph)", backend)
            self.assertIn("reserved_mib=%.2f plans=%zu replans=%llu", meta)
            self.assertNotIn("BIGCHERRY_META_ARENA_FASTBIND", meta)

    def test_production_set_and_flag_contract(self):
        with (_REPO / "config/recipes.toml").open("rb") as handle:
            recipes = tomllib.load(handle)
        production = recipes["patch-set"]["validated-enhancements"]["patches"]
        self.assertNotIn("1339_meta_memory_report", production)
        self.assertIn("1340_meta_per_device_arena", production)
        self.assertNotIn("meta-memory", recipes["experiment"])
        self.assertEqual([doc.name for doc in _MERGED.DOCUMENTED_ENV_DOCS], ["BIGCHERRY_META_MEM", "BIGCHERRY_META_PER_DEVICE_ARENA"])

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
