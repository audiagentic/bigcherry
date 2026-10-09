"""Offline mechanics tests for 1293_sched_single_input_sync at the pinned llama.cpp revision."""

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
_VENDOR = _REPO / "vendor/llama.cpp/ggml/src/ggml-backend.cpp"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_p1293 = _load("patch_1293", "engines/llamacpp/patches/1293_sched_single_input_sync/patch.py")
_p1326 = _load("patch_1326_for_1293", "engines/llamacpp/patches/1326_sched_async_host_inputs/patch.py")


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1293Mechanics(unittest.TestCase):
    def _copy_backend(self, root: Path) -> None:
        rel = "ggml/src/ggml-backend.cpp"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        copy_pinned(_REPO / "vendor/llama.cpp" / rel, root / rel)

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._copy_backend(root)
            first = apply_all(_p1293.PATCHES, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            text = (root / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
            self.assertIn("bool bc_user_inputs_synced = false;", text)
            self.assertIn("bool * user_inputs_synced", text)
            self.assertEqual(text.count("&bc_user_inputs_synced"), 2)
            self.assertIn("} else if (!*user_inputs_synced) {", text)
            before = text
            second = apply_all(_p1293.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, (root / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8"))

    def test_composes_before_production_1326(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._copy_backend(root)
            rel = "ggml/src/ggml-backend-meta.cpp"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_REPO / "vendor/llama.cpp" / rel, root / rel)
            r1293 = apply_all(_p1293.PATCHES, root)
            self.assertTrue(all(r.ok for r in r1293), [e.detail for r in r1293 for e in r.failed])
            r1326 = apply_all(_p1326.PATCHES, root)
            self.assertTrue(all(r.ok for r in r1326), [e.detail for r in r1326 for e in r.failed])
            text = (root / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
            self.assertLess(text.index("bigcherry 1326: async host->device input copy"),
                            text.index("if (input->flags & GGML_TENSOR_FLAG_INPUT) {"))

    def test_helper_anchor_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._copy_backend(root)
            path = root / "ggml/src/ggml-backend.cpp"
            text = path.read_text(encoding="utf-8")
            text = text.replace(
                "static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input) {",
                "static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, struct ggml_tensor * input) { // drift",
                1,
            )
            path.write_text(text, encoding="utf-8")
            results = apply_all(_p1293.PATCHES, root)
            self.assertTrue(any(not r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
