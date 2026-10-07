"""Offline mechanics tests for 1316_node_hash_trace on b11474 scheduler/context sources."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_VENDOR_CTX = _REPO / "vendor/llama.cpp/src/llama-context.cpp"
_VENDOR_BACKEND = _REPO / "vendor/llama.cpp/ggml/src/ggml-backend.cpp"
_spec = importlib.util.spec_from_file_location("patch_1316", _REPO / "patches/1316_node_hash_trace/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class Patch1316StaticContracts(unittest.TestCase):
    def test_meta_buffers_are_skipped_before_tensor_get(self):
        src = (_REPO / "patches/1316_node_hash_trace/patch.py").read_text(encoding="utf-8")
        self.assertIn("ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_META", src)
        self.assertIn("skip=meta", src)
        self.assertLess(src.index("if (bc_node_hash_is_meta_tensor(t))"), src.index("ggml_backend_tensor_get(t, buf.data(), 0, n);"))
        self.assertIn("t->view_src != nullptr ? t->view_src->buffer : t->buffer", src)

    def test_meta_scheduler_split_is_atomic(self):
        src = (_REPO / "patches/1316_node_hash_trace/patch.py").read_text(encoding="utf-8")
        self.assertIn("GGML_BACKEND_DEVICE_TYPE_META", src)
        self.assertIn("ggml_backend_graph_compute_async(split_backend, &split->graph)", src)
        self.assertIn("Meta owns its own subgraph partition/reduction walk", src)

    def test_value_format_is_from_count(self):
        src = (_REPO / "patches/1316_node_hash_trace/patch.py").read_text(encoding="utf-8")
        self.assertIn('std::sscanf(env, "%ld:%ld"', src)
        self.assertIn("must be from:count with from >= 0 and count > 0", src)
        self.assertEqual([doc.values for doc in _module.ENV_DOCS], ["<from>:<count>"])


@unittest.skipUnless(_VENDOR_CTX.exists() and _VENDOR_BACKEND.exists(), "pinned vendor checkout not present")
class Patch1316Mechanics(unittest.TestCase):
    def test_composes_after_production_scheduler_input_patch(self):
        prod = _REPO / "patches/1326_sched_async_host_inputs/patch.py"
        spec = importlib.util.spec_from_file_location("patch_1326_for_1316", prod)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "src/llama-context.cpp"
            backend_path = root / "ggml/src/ggml-backend.cpp"
            path.parent.mkdir(parents=True)
            backend_path.parent.mkdir(parents=True)
            copy_pinned(_VENDOR_CTX, path)
            copy_pinned(_VENDOR_BACKEND, backend_path)

            sched_only = tuple(p for p in mod.PATCHES if p.path == "ggml/src/ggml-backend.cpp")
            prod_res = apply_all(sched_only, root)
            self.assertTrue(all(r.ok for r in prod_res), [e.detail for r in prod_res for e in r.failed])

            res = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            backend = backend_path.read_text(encoding="utf-8")
            self.assertIn("bigcherry 1326: async host->device input copy", backend)
            self.assertIn("Meta owns its own subgraph partition/reduction walk", backend)

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "src/llama-context.cpp"
            backend_path = root / "ggml/src/ggml-backend.cpp"
            path.parent.mkdir(parents=True)
            backend_path.parent.mkdir(parents=True)
            copy_pinned(_VENDOR_CTX, path)
            copy_pinned(_VENDOR_BACKEND, backend_path)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            helpers = out.index("static bool bc_node_hash_cb(")
            ctor = out.index("cparams.cb_eval           = bc_node_hash_cb;")
            self.assertLess(helpers, ctor)
            self.assertLess(out.index("cparams.cb_eval_user_data = params.cb_eval_user_data;"), ctor)
            adv = out.index("bigcherry 1316: advance this context's graph index")
            self.assertLess(adv, out.index("auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);"))
            self.assertLess(out.index("ggml_status llama_context::graph_compute("), adv)
            self.assertIn("skip=meta", out)
            self.assertLess(out.index("if (bc_node_hash_is_meta_tensor(t))"), out.index("ggml_backend_tensor_get(t, buf.data(), 0, n);"))
            backend = backend_path.read_text(encoding="utf-8")
            self.assertIn("Meta owns its own subgraph partition/reduction walk", backend)
            self.assertIn("GGML_BACKEND_DEVICE_TYPE_META", backend)
            meta_compute = backend.index("ggml_backend_graph_compute_async(split_backend, &split->graph)")
            generic_view = backend.index("struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);")
            self.assertLess(meta_compute, generic_view)
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))
            self.assertEqual(backend, backend_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
