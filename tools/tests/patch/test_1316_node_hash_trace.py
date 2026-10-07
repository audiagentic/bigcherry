"""Offline mechanics tests for 1316_node_hash_trace (pinned src/llama-context.cpp)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/src/llama-context.cpp"
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

    def test_value_format_is_from_count(self):
        src = (_REPO / "patches/1316_node_hash_trace/patch.py").read_text(encoding="utf-8")
        self.assertIn('std::sscanf(env, "%ld:%ld"', src)
        self.assertIn("must be from:count with from >= 0 and count > 0", src)
        self.assertEqual([doc.values for doc in _module.ENV_DOCS], ["<from>:<count>"])


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1316Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "src/llama-context.cpp"
            path.parent.mkdir(parents=True)
            copy_pinned(_VENDOR, path)
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
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
