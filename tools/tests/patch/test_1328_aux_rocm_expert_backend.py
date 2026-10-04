"""Offline mechanics tests for 1328_aux_rocm_expert_backend and deploy-v5-plus-1327 composition."""

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
_RELS = (
    "ggml/include/ggml-backend.h",
    "ggml/src/ggml-backend-meta.cpp",
    "ggml/src/ggml-backend.cpp",
    "src/llama-context.cpp",
    "src/models/qwen4exp.cpp",
)
_DEPLOY = (
    "1291_ar_cpu_root",
    "1292_kpool_tail_truncate",
    "1294_topk_deterministic_ties",
    "1297_draft_vocab_trim",
    "1302_cuda_graph_oom_evict",
    "1303_attn_kv_tensor_split",
    "1235_rd09_q81_activation_cache_foundation",
    "1307_q81_activation_cache_mmvq",
    "1308_qwen4exp_rollback_copy_no_cont",
    "1309_rms_norm_mul_q81",
    "1310_act_q81",
    "1311_hc_pre_q81",
    "1312_mul_q81",
    "1313_scale_act_fuse",
    "1326_sched_async_host_inputs",
    "1327_qsa_host_remap",
)


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _only(mod):
    return [p for p in mod.PATCHES if p.path in _RELS]


_P1328 = _load("1328_aux_rocm_expert_backend")


@unittest.skipUnless(all((_REPO / "vendor/llama.cpp" / rel).exists() for rel in _RELS), "pinned vendor checkout not present")
class Patch1328Mechanics(unittest.TestCase):
    def _tree(self, td):
        root = Path(td)
        for rel in _RELS:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_REPO / "vendor/llama.cpp" / rel, dst)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            res = apply_all(_P1328.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            backend = (root / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
            meta = (root / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
            ctx = (root / "src/llama-context.cpp").read_text(encoding="utf-8")
            qwen = (root / "src/models/qwen4exp.cpp").read_text(encoding="utf-8")
            self.assertIn("BIGCHERRY_EXPERT_AUX_DEVICE", ctx)
            self.assertIn("named ordinary GPU through its pinned host buffer", backend)
            self.assertIn("BIGCHERRY_AUX_EXPERT_MERGE_MAGIC", meta)
            self.assertIn("partial auxiliary routed-expert placement is unsupported", qwen)
            self.assertIn("ggml_backend_meta_mark_mirrored_partial_add(cur)", qwen)

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
