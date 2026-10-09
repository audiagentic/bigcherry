"""Offline mechanics tests for 1328_aux_rocm_expert_backend and b11474 production-side scheduler composition."""

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
_RELS = (
    "ggml/include/ggml-backend.h",
    "ggml/src/ggml-backend-meta.cpp",
    "ggml/src/ggml-backend.cpp",
    "src/llama-context.cpp",
    "src/models/qwen4exp.cpp",
)
_POST_META = (
    "1340_meta_per_device_arena",
    "1341_meta_subset_mirrored",
)

_DEPLOY = (
    "1291_ar_cpu_root",
    "1292_kpool_tail_truncate",
    "1294_topk_deterministic_ties",
    "1297_draft_vocab_trim",
    "1302_cuda_graph_oom_evict",
    "1303_attn_kv_tensor_split",
    "1307_q81_activation_cache_mmvq",
    "1308_qwen4exp_rollback_copy_no_cont",
    "1313_scale_act_fuse",
    "1326_sched_async_host_inputs",
    "1327_qsa_host_remap",
)


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "engines" / "llamacpp" / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _only(mod):
    return [p for p in mod.PATCHES if p.path in _RELS]


_P1328 = _load("1328_aux_rocm_expert_backend")


@unittest.skipUnless(all((paths.llama_root() / rel).exists() for rel in _RELS), "pinned vendor checkout not present")
class Patch1328Mechanics(unittest.TestCase):
    def _tree(self, td):
        root = Path(td)
        for rel in _RELS:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(paths.llama_root() / rel, dst)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            res = apply_all(_P1328.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            header = (root / "ggml/include/ggml-backend.h").read_text(encoding="utf-8")
            backend = (root / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
            meta = (root / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
            ctx = (root / "src/llama-context.cpp").read_text(encoding="utf-8")
            qwen = (root / "src/models/qwen4exp.cpp").read_text(encoding="utf-8")
            self.assertIn("BIGCHERRY_EXPERT_AUX_DEVICE", ctx)
            self.assertIn("const enum ggml_backend_dev_type bc_aux_type", ctx)
            self.assertIn("scoped auxiliary expert backend", backend)
            self.assertIn("ggml_backend_sched_set_aux_expert_backend", header)
            self.assertIn("ggml_backend_sched_set_aux_expert_backend(sched.get(), bc_aux_backend)", ctx)
            self.assertIn("sched->bc_aux_backend != nullptr && b == sched->bc_aux_backend", backend)
            self.assertNotIn('const char * bc_aux_name = getenv("BIGCHERRY_EXPERT_AUX_DEVICE");\n    auto bc_backend_is_meta', backend)
            self.assertIn("BIGCHERRY_AUX_EXPERT_MERGE_MAGIC", meta)
            self.assertIn("mirrored_mirrored", meta)
            self.assertIn("MIRRORED + PARTIAL or MIRRORED + MIRRORED", meta)
            self.assertIn("partial auxiliary routed-expert placement is unsupported", qwen)
            self.assertIn("ggml_backend_meta_mark_mirrored_partial_add(cur)", qwen)
            self.assertIn("hook=qwen_ffn_entry", qwen)
            self.assertIn("hook=aux_merge_build", qwen)
            self.assertIn("hook=sched_copy", backend)
            self.assertIn("hook=sched_execute_merge", backend)
            self.assertNotIn("BIGCHERRY_PATCH_HIT patch=1328_aux_rocm_expert_backend", meta)
            self.assertGreaterEqual(backend.count('getenv("BIGCHERRY_PATCH_TRACE")'), 4)
            self.assertGreaterEqual(qwen.count('getenv("BIGCHERRY_PATCH_TRACE")'), 2)
            self.assertGreaterEqual(ctx.count('getenv("BIGCHERRY_PATCH_TRACE")'), 1)
            self.assertIn("ggml_backend_meta_is_mirrored_partial_add(split->graph.nodes[j])", backend)
            self.assertIn('hook=sched_assign_merge tensor=%s scoped_aux=%s\\n"', backend)
            self.assertIn('hook=sched_execute_merge split=%d node=%d backend=%s\\n"', backend)
            self.assertNotIn('hook=sched_assign_merge tensor=%s scoped_aux=%s\n"', backend)
            self.assertNotIn('hook=sched_execute_merge split=%d node=%d backend=%s\n"', backend)
            self.assertIn('hook=aux_merge_build layer=%d tokens=%lld ctx_type=%d\\n"', qwen)
            self.assertNotIn('hook=aux_merge_build layer=%d tokens=%lld ctx_type=%d\n"', qwen)
            self.assertIn("cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT", qwen)
            self.assertIn("cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT", ctx)

            before = {rel: (root / rel).read_bytes() for rel in _RELS}
            second = apply_all(_P1328.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, {rel: (root / rel).read_bytes() for rel in _RELS})

    def test_composes_after_deploy_v5_plus_1327_same_file_edits(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            touched = []
            for pid in sorted(_DEPLOY, key=lambda x: int(x.split("_", 1)[0])):
                patches = _only(_load(pid))
                if not patches:
                    continue
                touched.append(pid)
                res = apply_all(patches, root)
                self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))
            self.assertIn("1326_sched_async_host_inputs", touched)
            self.assertIn("1327_qsa_host_remap", touched)
            res = apply_all(_P1328.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

    def test_composes_with_current_production_meta_patches(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            for pid in sorted(_DEPLOY, key=lambda x: int(x.split("_", 1)[0])):
                patches = _only(_load(pid))
                if patches:
                    res = apply_all(patches, root)
                    self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))

            res = apply_all(_P1328.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            for pid in _POST_META:
                patches = _only(_load(pid))
                if patches:
                    res = apply_all(patches, root)
                    self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))

            meta = (root / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
            self.assertIn("mirrored_mirrored", meta)
            self.assertIn("active_mask", meta)

    def test_split_state_cache_reuse_is_linear_for_stacked_marked_adds(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            for pid in sorted(_DEPLOY, key=lambda x: int(x.split("_", 1)[0])):
                patches = _only(_load(pid))
                if patches:
                    res = apply_all(patches, root)
                    self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))

            res = apply_all(_P1328.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            for pid in _POST_META:
                patches = _only(_load(pid))
                if patches:
                    res = apply_all(patches, root)
                    self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))

            meta = (root / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
            cache_block = meta[meta.index("const std::pair key = std::make_pair(tensor, assume_sync);"):]
            cache_block = cache_block[:cache_block.index("ggml_backend_meta_split_state ret =")]
            self.assertIn("buf_ctx->split_state_cache.erase(it);", cache_block)
            self.assertNotIn("buf_ctx->split_state_cache.clear();", cache_block)

            # Model sequential Meta buffer-init queries for N stacked marked adds. Each new graph tensor reuses
            # an address with one stale prior-graph snapshot. Per-key eviction preserves already-computed current
            # ancestors; whole-cache invalidation would make the total recursive evaluations 1+2+...+N.
            n = 32
            mirrored = object()
            nodes = []
            prev = mirrored
            for i in range(n):
                node = ("marked_add", i, prev, mirrored)
                nodes.append(node)
                prev = node

            cache = {}
            evaluations = 0

            def split_state(node):
                nonlocal evaluations
                if node is mirrored:
                    return "MIRRORED"
                key = node[1]
                snapshot = ("current", key)
                entry = cache.get(key)
                if entry is not None and entry[1] != snapshot:
                    cache.pop(key)
                    entry = None
                if entry is not None:
                    return entry[0]

                evaluations += 1
                left = split_state(node[2])
                right = split_state(node[3])
                self.assertEqual((left, right), ("MIRRORED", "MIRRORED"))
                cache[key] = ("MIRRORED", snapshot)
                return "MIRRORED"

            for node in nodes:
                key = node[1]
                cache[key] = ("MIRRORED", ("stale", key))
                self.assertEqual(split_state(node), "MIRRORED")

            self.assertLessEqual(evaluations, n + 1)

    def test_missing_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            path = root / "src/llama-context.cpp"
            text = path.read_text(encoding="utf-8").replace("        // GPU backends\n", "        // changed upstream\n", 1)
            path.write_text(text, encoding="utf-8", newline="")
            res = apply_all(_P1328.PATCHES, root)
            failures = [e for r in res for e in r.failed]
            self.assertTrue(failures)
            self.assertTrue(any("target-aux-backend" in e.edit_id for e in failures))


if __name__ == "__main__":
    unittest.main()
