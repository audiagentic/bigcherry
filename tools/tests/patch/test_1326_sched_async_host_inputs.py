"""Offline mechanics tests for 1326_sched_async_host_inputs (pinned ggml/src/ggml-backend.cpp)."""

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
_VENDOR = paths.llama_root() / "ggml/src/ggml-backend.cpp"
_spec = importlib.util.spec_from_file_location("patch_1326", _REPO / "engines/llamacpp/patches/1326_sched_async_host_inputs/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1326Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ("ggml/src/ggml-backend.cpp", "ggml/src/ggml-backend-meta.cpp"):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                copy_pinned(paths.llama_root() / rel, root / rel)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            be = (root / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
            hook = be.index("bigcherry 1326: async host->device input copy")
            self.assertLess(be.index("struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);"), hook)
            self.assertLess(hook, be.index("if (input->flags & GGML_TENSOR_FLAG_INPUT) {", hook))
            self.assertIn("memcpy(bc_stage.data(), input->data, ggml_nbytes(input));", be)
            meta = (root / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
            fn = meta.index("static void ggml_backend_meta_set_tensor_async(")
            fb = meta.index("bigcherry 1326: states the async splice cannot express")
            self.assertLess(fn, fb)
            self.assertLess(fb, meta.index("static void ggml_backend_meta_get_tensor_async("))
            before = (be, meta)
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, ((root / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8"),
                                      (root / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
